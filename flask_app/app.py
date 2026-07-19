"""Flask GUI for browsing and cropping the Rubin Observatory "Ocean of Stars" image.

Two sources are used:
- a small "low-res" JPEG (bundled in the repo) used for the on-screen preview map
- an optional full-resolution 16-bit TIFF (56428 x 29949 px, ~10 GB) that the user
  downloads separately, opened via a numpy memmap so cropping never loads the
  whole file into RAM.
"""
import logging
import math
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import tifffile
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

logging.getLogger("tifffile").setLevel(logging.ERROR)

# Plate solution. The tangent point (RA0/DEC0) is the image's published field
# center. An initial fit using ~150 bright (G<13) stars gave a good *average*
# fit (median ~1px residual) but was locally biased by several pixels in
# under-sampled parts of the frame - a rigid (rotation+scale+translation)
# transform fit from too few, too-sparse points can't average out local noise.
# The fit below instead uses ~4500 stars (G<14.5) spanning the entire frame,
# matched to their detected pixel peaks, with a degree-3 polynomial distortion
# term on top of the gnomonic/TAN projection (silimar to a SIP-distorted WCS).
# Residual after fitting: median ~0.4 px, max ~2.5 px across the whole frame
# (see flask_app/tools/plate_solve.py).
RA0_DEG = 224.76984640789192
DEC0_DEG = -37.939446521397656
_RA0 = math.radians(RA0_DEG)
_DEC0 = math.radians(DEC0_DEG)

# xi/eta (gnomonic radians) -> pixel, via a degree-3 polynomial in (xi, eta):
# basis = [1, xi, eta, xi^2, xi*eta, eta^2, xi^3, xi^2*eta, xi*eta^2, eta^3]
# px = PLATE_CX . basis   py = PLATE_CY . basis
PLATE_CX = (
    1999.7636315234488, -73088.16239545682, 97.47720777489019,
    -2427.235450950923, -4.351627915670664, -169.98195207141725,
    86835.41713612381, 1921.6427119784203, 676.4196286585695, -18562.04584138326,
)
PLATE_CY = (
    1060.9556173595986, -98.86461543816775, -73115.4770637289,
    -1220.0426883027787, -885.956838531343, 18.552137270575322,
    -8218.147610689712, 102.28652799927075, 1529.3802893802067, 8569.652326636984,
)
PLATE_ARCSEC_PER_PX = 206264.80624709636 / math.hypot(PLATE_CX[1], PLATE_CX[2])

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
LIGHT_YEARS_PER_PARSEC = 3.26156
GAIA_EPOCH = 2016.0    # gaiadr3.gaia_source reference epoch (Julian year)
IMAGE_EPOCH = 2025.5   # approximate observation epoch of the Ocean of Stars image

BASE_DIR = Path(__file__).resolve().parent
MASTER_IMAGE_PATH = BASE_DIR / "static" / "img" / "ocean_of_stars.jpg"

# Full-resolution TIFF downloaded separately (see README) - not part of the repo.
HIRES_TIFF_PATH = Path(
    os.environ.get("OCEAN_OF_STARS_TIFF", r"C:\Users\thoma\Downloads\noirlab2616a.tif")
)

# Cap on the largest dimension we'll ever materialize/send for a hi-res crop,
# so a huge drag selection can't try to read gigabytes at once.
MAX_HIRES_OUTPUT_DIM = 4000

app = Flask(__name__)

_master_image = Image.open(MASTER_IMAGE_PATH)
_master_image.load()
MASTER_WIDTH, MASTER_HEIGHT = _master_image.size

_hires_mmap = None
HIRES_WIDTH = HIRES_HEIGHT = None
HIRES_AVAILABLE = False

if HIRES_TIFF_PATH.exists():
    try:
        _hires_mmap = tifffile.memmap(str(HIRES_TIFF_PATH))
        HIRES_HEIGHT, HIRES_WIDTH = _hires_mmap.shape[0], _hires_mmap.shape[1]
        HIRES_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001 - report and keep running without hi-res
        print(f"Could not open hi-res TIFF at {HIRES_TIFF_PATH}: {exc}")
else:
    print(f"Hi-res TIFF not found at {HIRES_TIFF_PATH} - high-res tab will be disabled.")

IMAGE_INFO = {
    "width": MASTER_WIDTH,
    "height": MASTER_HEIGHT,
    "title": "Ocean of Stars",
    "credit": "NSF-DOE Vera C. Rubin Observatory/NOIRLab/SLAC/AURA",
    "source_url": "https://noirlab.edu/public/images/noirlab2616a/",
    "field_of_view": "188.33 x 99.95 arcminutes",
    "center_ra": "14h 59m 04.76s",
    "center_dec": "-37 56 22.01",
    "hires_available": HIRES_AVAILABLE,
    "hires_width": HIRES_WIDTH,
    "hires_height": HIRES_HEIGHT,
    "hires_path": str(HIRES_TIFF_PATH),
}


@app.route("/")
def index():
    return render_template("index.html", info=IMAGE_INFO)


@app.route("/api/info")
def api_info():
    return jsonify(IMAGE_INFO)


def _gnomonic_forward(ra_deg, dec_deg):
    """Standard TAN/gnomonic projection: (ra_deg, dec_deg) -> (xi, eta) in radians."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    dra = ra - _RA0
    denom = math.sin(_DEC0) * math.sin(dec) + math.cos(_DEC0) * math.cos(dec) * math.cos(dra)
    xi = math.cos(dec) * math.sin(dra) / denom
    eta = (math.cos(_DEC0) * math.sin(dec) - math.sin(_DEC0) * math.cos(dec) * math.cos(dra)) / denom
    return xi, eta


def _gnomonic_inverse(xi, eta):
    """Inverse of _gnomonic_forward: (xi, eta) radians -> (ra_deg, dec_deg)."""
    rho = math.hypot(xi, eta)
    if rho < 1e-12:
        return RA0_DEG, DEC0_DEG
    c = math.atan(rho)
    sin_c, cos_c = math.sin(c), math.cos(c)
    dec = math.asin(cos_c * math.sin(_DEC0) + (eta * sin_c * math.cos(_DEC0)) / rho)
    ra = _RA0 + math.atan2(xi * sin_c, rho * math.cos(_DEC0) * cos_c - eta * math.sin(_DEC0) * sin_c)
    return math.degrees(ra), math.degrees(dec)


def _poly_basis(xi, eta):
    return (1.0, xi, eta, xi * xi, xi * eta, eta * eta,
            xi ** 3, xi * xi * eta, xi * eta * eta, eta ** 3)


def _poly_basis_deriv(xi, eta):
    """d(basis)/dxi, d(basis)/deta - for Newton's-method inversion."""
    d_dxi = (0.0, 1.0, 0.0, 2 * xi, eta, 0.0, 3 * xi * xi, 2 * xi * eta, eta * eta, 0.0)
    d_deta = (0.0, 0.0, 1.0, 0.0, xi, 2 * eta, 0.0, xi * xi, 2 * xi * eta, 3 * eta * eta)
    return d_dxi, d_deta


def radec_to_lowres_pixel(ra, dec):
    """Plate-solved conversion, (ra_deg, dec_deg) -> low-res preview pixel."""
    xi, eta = _gnomonic_forward(ra, dec)
    basis = _poly_basis(xi, eta)
    px = sum(c * b for c, b in zip(PLATE_CX, basis))
    py = sum(c * b for c, b in zip(PLATE_CY, basis))
    return px, py


def lowres_pixel_to_radec(px, py):
    """Inverse of radec_to_lowres_pixel, via Newton's method (the polynomial
    distortion has no closed-form inverse). Converges in a couple of
    iterations since distortion is a small correction on the dominant
    linear term."""
    a, b, c, d = PLATE_CX[1], PLATE_CX[2], PLATE_CY[1], PLATE_CY[2]
    det = a * d - b * c
    dx0, dy0 = px - PLATE_CX[0], py - PLATE_CY[0]
    xi = (d * dx0 - b * dy0) / det
    eta = (-c * dx0 + a * dy0) / det

    for _ in range(8):
        basis = _poly_basis(xi, eta)
        fx = sum(cc * bb for cc, bb in zip(PLATE_CX, basis)) - px
        fy = sum(cc * bb for cc, bb in zip(PLATE_CY, basis)) - py
        d_dxi, d_deta = _poly_basis_deriv(xi, eta)
        dpx_dxi = sum(cc * bb for cc, bb in zip(PLATE_CX, d_dxi))
        dpx_deta = sum(cc * bb for cc, bb in zip(PLATE_CX, d_deta))
        dpy_dxi = sum(cc * bb for cc, bb in zip(PLATE_CY, d_dxi))
        dpy_deta = sum(cc * bb for cc, bb in zip(PLATE_CY, d_deta))
        det_j = dpx_dxi * dpy_deta - dpx_deta * dpy_dxi
        if abs(det_j) < 1e-30:
            break
        delta_xi = (dpx_deta * fy - dpy_deta * fx) / det_j
        delta_eta = (dpy_dxi * fx - dpx_dxi * fy) / det_j
        xi += delta_xi
        eta += delta_eta
        if abs(delta_xi) < 1e-14 and abs(delta_eta) < 1e-14:
            break
    return _gnomonic_inverse(xi, eta)


# Approximate main-sequence temperature/color -> spectral class boundaries.
# Gaia doesn't do formal MK classification, so this is a rough estimate from
# teff_gspphot (preferred) or, when that's unavailable, BP-RP color.
_TEFF_BOUNDARIES = [(30000, "O"), (10000, "B"), (7500, "A"), (6000, "F"), (5200, "G"), (3700, "K")]
_BP_RP_BOUNDARIES = [(-0.1, "O"), (0.0, "B"), (0.32, "A"), (0.56, "F"), (0.85, "G"), (1.40, "K")]


def estimate_spectral_type(teff, bp_rp):
    if teff is not None:
        for lower, cls in _TEFF_BOUNDARIES:
            if teff >= lower:
                return cls
        return "M"
    if bp_rp is not None:
        for upper, cls in _BP_RP_BOUNDARIES:
            if bp_rp <= upper:
                return cls
        return "M"
    return None


def format_ra(ra_deg):
    h = ra_deg / 15.0
    hh = int(h)
    mm_f = (h - hh) * 60.0
    mm = int(mm_f)
    ss = (mm_f - mm) * 60.0
    return f"{hh:02d}h {mm:02d}m {ss:05.2f}s"


def format_dec(dec_deg):
    sign = "-" if dec_deg < 0 else "+"
    d = abs(dec_deg)
    dd = int(d)
    mm_f = (d - dd) * 60.0
    mm = int(mm_f)
    ss = (mm_f - mm) * 60.0
    return f"{sign}{dd:02d}° {mm:02d}' {ss:04.1f}\""


def _parse_box(args, max_w, max_h):
    x = int(args.get("x", 0))
    y = int(args.get("y", 0))
    w = int(args.get("w", 0))
    h = int(args.get("h", 0))
    if w <= 0 or h <= 0:
        raise ValueError("w and h must be positive")

    x0 = max(0, min(x, max_w - 1))
    y0 = max(0, min(y, max_h - 1))
    x1 = max(x0 + 1, min(x + w, max_w))
    y1 = max(y0 + 1, min(y + h, max_h))
    return x0, y0, x1, y1


@app.route("/api/crop")
def api_crop():
    try:
        x0, y0, x1, y1 = _parse_box(request.args, MASTER_WIDTH, MASTER_HEIGHT)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    crop = _master_image.crop((x0, y0, x1, y1))

    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.headers["X-Crop-X"] = str(x0)
    resp.headers["X-Crop-Y"] = str(y0)
    resp.headers["X-Output-Width"] = str(x1 - x0)
    resp.headers["X-Output-Height"] = str(y1 - y0)
    return resp


@app.route("/api/crop_hires")
def api_crop_hires():
    if not HIRES_AVAILABLE:
        return jsonify({
            "error": f"Hi-res TIFF not available on the server (expected at {HIRES_TIFF_PATH})."
        }), 503

    # Incoming x/y/w/h are in low-res preview pixel space; scale to the full TIFF.
    try:
        lx0, ly0, lx1, ly1 = _parse_box(request.args, MASTER_WIDTH, MASTER_HEIGHT)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scale_x = HIRES_WIDTH / MASTER_WIDTH
    scale_y = HIRES_HEIGHT / MASTER_HEIGHT

    x0 = max(0, min(int(lx0 * scale_x), HIRES_WIDTH - 1))
    y0 = max(0, min(int(ly0 * scale_y), HIRES_HEIGHT - 1))
    x1 = max(x0 + 1, min(int(lx1 * scale_x), HIRES_WIDTH))
    y1 = max(y0 + 1, min(int(ly1 * scale_y), HIRES_HEIGHT))

    native_w, native_h = x1 - x0, y1 - y0
    step = max(1, math.ceil(max(native_w, native_h) / MAX_HIRES_OUTPUT_DIM))

    region = np.array(_hires_mmap[y0:y1:step, x0:x1:step, :])  # forces the mmap read
    region_8bit = (region >> 8).astype(np.uint8)

    crop = Image.fromarray(region_8bit, "RGB")

    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.headers["X-Native-X"] = str(x0)
    resp.headers["X-Native-Y"] = str(y0)
    resp.headers["X-Native-Width"] = str(native_w)
    resp.headers["X-Native-Height"] = str(native_h)
    resp.headers["X-Output-Width"] = str(crop.width)
    resp.headers["X-Output-Height"] = str(crop.height)
    resp.headers["X-Downsample-Step"] = str(step)
    return resp


@app.route("/api/stars")
def api_stars():
    try:
        x0, y0, x1, y1 = _parse_box(request.args, MASTER_WIDTH, MASTER_HEIGHT)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    radec_corners = [lowres_pixel_to_radec(px, py) for px, py in corners]
    ras = [c[0] for c in radec_corners]
    decs = [c[1] for c in radec_corners]
    ra_min, ra_max = min(ras), max(ras)
    dec_min, dec_max = min(decs), max(decs)
    center_ra, center_dec = lowres_pixel_to_radec((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    sky_region = {
        "ra_min": ra_min, "ra_max": ra_max,
        "dec_min": dec_min, "dec_max": dec_max,
        "center_ra": center_ra, "center_dec": center_dec,
        "center_ra_str": format_ra(center_ra), "center_dec_str": format_dec(center_dec),
        "ra_min_str": format_ra(ra_min), "ra_max_str": format_ra(ra_max),
        "dec_min_str": format_dec(dec_min), "dec_max_str": format_dec(dec_max),
        "width_arcmin": (x1 - x0) * PLATE_ARCSEC_PER_PX / 60.0,
        "height_arcmin": (y1 - y0) * PLATE_ARCSEC_PER_PX / 60.0,
        "lowres_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }

    if HIRES_AVAILABLE:
        scale_x = HIRES_WIDTH / MASTER_WIDTH
        scale_y = HIRES_HEIGHT / MASTER_HEIGHT
        # Mirror the int-floored bounds crop_hires actually extracts, so marker
        # fractions computed against this box line up with the delivered pixels.
        hx0 = max(0, min(int(x0 * scale_x), HIRES_WIDTH - 1))
        hy0 = max(0, min(int(y0 * scale_y), HIRES_HEIGHT - 1))
        hx1 = max(hx0 + 1, min(int(x1 * scale_x), HIRES_WIDTH))
        hy1 = max(hy0 + 1, min(int(y1 * scale_y), HIRES_HEIGHT))
        sky_region["hires_box"] = {"x0": hx0, "y0": hy0, "x1": hx1, "y1": hy1}

    adql = (
        "SELECT TOP 10 source_id, ra, dec, phot_g_mean_mag, pmra, pmdec, teff_gspphot, bp_rp, parallax "
        "FROM gaiadr3.gaia_source "
        f"WHERE ra BETWEEN {ra_min!r} AND {ra_max!r} "
        f"AND dec BETWEEN {dec_min!r} AND {dec_max!r} "
        "AND phot_g_mean_mag IS NOT NULL "
        "ORDER BY phot_g_mean_mag ASC"
    )

    stars = []
    query_error = None
    try:
        resp = requests.get(
            GAIA_TAP_URL,
            params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        for row in payload.get("data", []):
            source_id, ra, dec, mag, pmra, pmdec, teff, bp_rp, parallax = row
            # Gaia DR3 positions are for epoch J2016.0; the image was taken
            # ~2025, so propagate proper motion forward before locating the
            # star in the image (up to ~0.5 arcsec for fast movers).
            ra_img, dec_img = ra, dec
            if pmra is not None and pmdec is not None:
                dt = IMAGE_EPOCH - GAIA_EPOCH
                dec_img = dec + (pmdec * dt / 3.6e6)
                ra_img = ra + (pmra * dt / 3.6e6) / math.cos(math.radians(dec))
            lx, ly = radec_to_lowres_pixel(ra_img, dec_img)
            star = {
                "source_id": source_id,
                "designation": f"Gaia DR3 {source_id}",
                "ra": ra, "dec": dec, "mag": mag,
                "ra_str": format_ra(ra), "dec_str": format_dec(dec),
                # +0.5: continuous pixel-center coordinate, so markers scale
                # correctly between resolutions (see hires_x below).
                "lowres_x": lx + 0.5, "lowres_y": ly + 0.5,
                "spectral_type": estimate_spectral_type(teff, bp_rp),
                "teff": teff,
            }
            if parallax is not None and parallax > 0:
                star["distance_ly"] = LIGHT_YEARS_PER_PARSEC * 1000.0 / parallax
            if pmra is not None and pmdec is not None:
                star["pm_mas_yr"] = math.hypot(pmra, pmdec)
                star["pm_direction_deg"] = math.degrees(math.atan2(pmra, pmdec)) % 360.0
            if HIRES_AVAILABLE:
                # Pixel centers map between resolutions as (i + 0.5) * scale, not
                # i * scale - the naive form is off by 0.5*scale - 0.5 = ~6.5 hires
                # px (~1.4 arcsec), a constant shift that's invisible at lowres but
                # obvious when a small selection is zoomed.
                star["hires_x"] = (lx + 0.5) * (HIRES_WIDTH / MASTER_WIDTH)
                star["hires_y"] = (ly + 0.5) * (HIRES_HEIGHT / MASTER_HEIGHT)
            stars.append(star)
    except requests.RequestException as exc:
        query_error = f"Could not reach the Gaia archive: {exc}"
    except (ValueError, KeyError) as exc:
        query_error = f"Unexpected response from the Gaia archive: {exc}"

    return jsonify({"sky_region": sky_region, "stars": stars, "query_error": query_error})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""Flask GUI for browsing and cropping Rubin Observatory images.

The registered images (see images.py) are NSF-DOE Vera C. Rubin Observatory
LSSTCam releases. Each has two sources:
- a small "low-res" JPEG used for the on-screen preview map
- an optional full-resolution 16-bit TIFF (tens of thousands of px per side,
  ~10 GB) opened via a numpy memmap so cropping never loads the whole file
  into RAM.

Neither file needs to be present on disk at startup - both can be fetched
on demand from the UI (see images.start_download).
"""
import logging
import math
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
import requests
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image

from images import IMAGE_STATES, IMAGES, MAX_HIRES_OUTPUT_DIM, get_download_status, start_download

logging.getLogger("tifffile").setLevel(logging.ERROR)

# Primary: an IVOA Cone Search mirror of Gaia DR3 (VOTable XML). Fallback:
# ESA's own TAP sync endpoint (JSON). The cone search mirror is used first -
# some networks (corporate proxies/firewalls) can reach the rest of
# gea.esac.esa.int fine but silently hang on requests to its TAP sync path
# specifically, while the plain HTTPS cone search mirror goes through.
GAIA_CONE_URL = "https://gaia.ari.uni-heidelberg.de/cone/search"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
LIGHT_YEARS_PER_PARSEC = 3.26156
GAIA_EPOCH = 2016.0  # gaiadr3.gaia_source reference epoch (Julian year)
DEFAULT_MAX_STARS = 100
MAX_STARS_LIMIT = 500  # cap so a mistyped/malicious value can't force a huge query

app = Flask(__name__)


def _get_state(key):
    state = IMAGE_STATES.get(key)
    if state is None:
        abort(404, description=f"Unknown image {key!r}. Known images: {sorted(IMAGES)}")
    return state


def _require_lowres(state):
    if not state.lowres_available:
        return jsonify({
            "error": f"The low-res preview for \"{state.cfg.title}\" hasn't been downloaded yet.",
            "needs_download": "lowres",
        }), 503
    return None


@app.route("/")
def index():
    return render_template("index.html")


def _image_summary(state):
    cfg = state.cfg
    return {
        "key": cfg.key,
        "title": cfg.title,
        "credit": cfg.credit,
        "source_url": cfg.source_url,
        "field_of_view": f"{cfg.field_of_view_arcmin[0]:.2f} x {cfg.field_of_view_arcmin[1]:.2f} arcminutes",
        "center_ra": format_ra(cfg.center_ra_deg),
        "center_dec": format_dec(cfg.center_dec_deg),
        "lowres_available": state.lowres_available,
        "lowres_filename": cfg.lowres_filename,
        "lowres_width": state.lowres_width,
        "lowres_height": state.lowres_height,
        "lowres_size_hint_bytes": cfg.lowres_size_hint_bytes,
        "hires_available": state.hires_available,
        "hires_width": state.hires_width,
        "hires_height": state.hires_height,
        "hires_size_hint_bytes": cfg.hires_size_hint_bytes,
        "hires_path": str(cfg.hires_path),
        "lowres_download_status": get_download_status(cfg.key, "lowres"),
        "hires_download_status": get_download_status(cfg.key, "hires"),
    }


@app.route("/api/images")
def api_images():
    return jsonify([_image_summary(state) for state in IMAGE_STATES.values()])


@app.route("/api/info")
def api_info():
    state = _get_state(request.args.get("image", ""))
    return jsonify(_image_summary(state))


@app.route("/api/download/<key>/<which>", methods=["POST"])
def api_download_start(key, which):
    if which not in ("lowres", "hires"):
        return jsonify({"error": "which must be 'lowres' or 'hires'"}), 400
    state = _get_state(key)
    if (which == "lowres" and state.lowres_available) or (which == "hires" and state.hires_available):
        return jsonify({"error": "already downloaded"}), 409
    started, message = start_download(key, which)
    if not started and message != "already downloading":
        return jsonify({"error": message}), 400
    return jsonify({"status": "started"})


@app.route("/api/download/<key>/<which>/status")
def api_download_status(key, which):
    _get_state(key)  # validates key
    return jsonify(get_download_status(key, which))


def _gnomonic_forward(ra_deg, dec_deg, ra0_rad, dec0_rad):
    """Standard TAN/gnomonic projection: (ra_deg, dec_deg) -> (xi, eta) in radians."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    dra = ra - ra0_rad
    denom = math.sin(dec0_rad) * math.sin(dec) + math.cos(dec0_rad) * math.cos(dec) * math.cos(dra)
    xi = math.cos(dec) * math.sin(dra) / denom
    eta = (math.cos(dec0_rad) * math.sin(dec) - math.sin(dec0_rad) * math.cos(dec) * math.cos(dra)) / denom
    return xi, eta


def _gnomonic_inverse(xi, eta, ra0_rad, dec0_rad):
    """Inverse of _gnomonic_forward: (xi, eta) radians -> (ra_deg, dec_deg)."""
    rho = math.hypot(xi, eta)
    if rho < 1e-12:
        return math.degrees(ra0_rad), math.degrees(dec0_rad)
    c = math.atan(rho)
    sin_c, cos_c = math.sin(c), math.cos(c)
    dec = math.asin(cos_c * math.sin(dec0_rad) + (eta * sin_c * math.cos(dec0_rad)) / rho)
    ra = ra0_rad + math.atan2(xi * sin_c, rho * math.cos(dec0_rad) * cos_c - eta * math.sin(dec0_rad) * sin_c)
    return math.degrees(ra), math.degrees(dec)


def _poly_basis(xi, eta):
    return (1.0, xi, eta, xi * xi, xi * eta, eta * eta,
            xi ** 3, xi * xi * eta, xi * eta * eta, eta ** 3)


def _poly_basis_deriv(xi, eta):
    """d(basis)/dxi, d(basis)/deta - for Newton's-method inversion."""
    d_dxi = (0.0, 1.0, 0.0, 2 * xi, eta, 0.0, 3 * xi * xi, 2 * xi * eta, eta * eta, 0.0)
    d_deta = (0.0, 0.0, 1.0, 0.0, xi, 2 * eta, 0.0, xi * xi, 2 * xi * eta, 3 * eta * eta)
    return d_dxi, d_deta


def radec_to_lowres_pixel(ra, dec, ra0_rad, dec0_rad, plate_cx, plate_cy):
    """Plate-solved conversion, (ra_deg, dec_deg) -> low-res preview pixel."""
    xi, eta = _gnomonic_forward(ra, dec, ra0_rad, dec0_rad)
    basis = _poly_basis(xi, eta)
    px = sum(c * b for c, b in zip(plate_cx, basis))
    py = sum(c * b for c, b in zip(plate_cy, basis))
    return px, py


def lowres_pixel_to_radec(px, py, ra0_rad, dec0_rad, plate_cx, plate_cy):
    """Inverse of radec_to_lowres_pixel, via Newton's method (the polynomial
    distortion has no closed-form inverse). Converges in a couple of
    iterations since distortion is a small correction on the dominant
    linear term."""
    a, b, c, d = plate_cx[1], plate_cx[2], plate_cy[1], plate_cy[2]
    det = a * d - b * c
    dx0, dy0 = px - plate_cx[0], py - plate_cy[0]
    xi = (d * dx0 - b * dy0) / det
    eta = (-c * dx0 + a * dy0) / det

    for _ in range(8):
        basis = _poly_basis(xi, eta)
        fx = sum(cc * bb for cc, bb in zip(plate_cx, basis)) - px
        fy = sum(cc * bb for cc, bb in zip(plate_cy, basis)) - py
        d_dxi, d_deta = _poly_basis_deriv(xi, eta)
        dpx_dxi = sum(cc * bb for cc, bb in zip(plate_cx, d_dxi))
        dpx_deta = sum(cc * bb for cc, bb in zip(plate_cx, d_deta))
        dpy_dxi = sum(cc * bb for cc, bb in zip(plate_cy, d_dxi))
        dpy_deta = sum(cc * bb for cc, bb in zip(plate_cy, d_deta))
        det_j = dpx_dxi * dpy_deta - dpx_deta * dpy_dxi
        if abs(det_j) < 1e-30:
            break
        delta_xi = (dpx_deta * fy - dpy_deta * fx) / det_j
        delta_eta = (dpy_dxi * fx - dpx_dxi * fy) / det_j
        xi += delta_xi
        eta += delta_eta
        if abs(delta_xi) < 1e-14 and abs(delta_eta) < 1e-14:
            break
    return _gnomonic_inverse(xi, eta, ra0_rad, dec0_rad)


def _plate_params(state):
    """(ra0_rad, dec0_rad, plate_cx, plate_cy) for this image's current plate solution."""
    plate_cx, plate_cy = state.plate_solution()
    return math.radians(state.cfg.center_ra_deg), math.radians(state.cfg.center_dec_deg), plate_cx, plate_cy


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
    state = _get_state(request.args.get("image", ""))
    missing = _require_lowres(state)
    if missing:
        return missing

    try:
        x0, y0, x1, y1 = _parse_box(request.args, state.lowres_width, state.lowres_height)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    crop = state.lowres_image.crop((x0, y0, x1, y1))

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
    state = _get_state(request.args.get("image", ""))
    missing = _require_lowres(state)
    if missing:
        return missing
    if not state.hires_available:
        return jsonify({
            "error": f"Hi-res TIFF not available on the server (expected at {state.cfg.hires_path}).",
            "needs_download": "hires",
        }), 503

    # Incoming x/y/w/h are in low-res preview pixel space; scale to the full TIFF.
    try:
        lx0, ly0, lx1, ly1 = _parse_box(request.args, state.lowres_width, state.lowres_height)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scale_x = state.hires_width / state.lowres_width
    scale_y = state.hires_height / state.lowres_height

    x0 = max(0, min(int(lx0 * scale_x), state.hires_width - 1))
    y0 = max(0, min(int(ly0 * scale_y), state.hires_height - 1))
    x1 = max(x0 + 1, min(int(lx1 * scale_x), state.hires_width))
    y1 = max(y0 + 1, min(int(ly1 * scale_y), state.hires_height))

    native_w, native_h = x1 - x0, y1 - y0
    step = max(1, math.ceil(max(native_w, native_h) / MAX_HIRES_OUTPUT_DIM))

    region = np.array(state.hires_mmap[y0:y1:step, x0:x1:step, :])  # forces the mmap read
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


_GAIA_COLUMNS = ("source_id", "ra", "dec", "phot_g_mean_mag", "pmra", "pmdec", "teff_gspphot", "bp_rp", "parallax")


def _votable_local(tag):
    """Strip the VOTable XML namespace off an ElementTree tag, e.g.
    '{http://www.ivoa.net/xml/VOTable/v1.3}FIELD' -> 'FIELD'."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _parse_votable_rows(xml_bytes, wanted_fields):
    root = ET.fromstring(xml_bytes)
    field_names = [el.get("name") for el in root.iter() if _votable_local(el.tag) == "FIELD"]
    rows = []
    for tr in root.iter():
        if _votable_local(tr.tag) != "TR":
            continue
        values = [td.text for td in tr if _votable_local(td.tag) == "TD"]
        row = dict(zip(field_names, values))
        rows.append({k: row.get(k) for k in wanted_fields})
    return rows


def _to_float(v):
    if v is None or v == "":
        return None
    return float(v)


def _query_gaia_cone(ra_min, ra_max, dec_min, dec_max, center_ra, center_dec, max_stars):
    """Query Gaia DR3 via an IVOA Cone Search mirror (circle around the
    selection's center, radius sized to cover its corners), then clip the
    results down to the actual selection box and keep the max_stars brightest."""
    dra = (ra_max - ra_min) / 2.0 * math.cos(math.radians(center_dec))
    ddec = (dec_max - dec_min) / 2.0
    radius_deg = math.hypot(dra, ddec) * 1.05 + 0.001  # pad past the box corners

    resp = requests.get(
        GAIA_CONE_URL,
        params={"RA": center_ra, "DEC": center_dec, "SR": radius_deg, "RESPONSEFORMAT": "votable"},
        timeout=25,
    )
    resp.raise_for_status()
    rows = _parse_votable_rows(resp.content, _GAIA_COLUMNS)

    out = []
    for row in rows:
        ra, dec = _to_float(row.get("ra")), _to_float(row.get("dec"))
        mag = _to_float(row.get("phot_g_mean_mag"))
        if ra is None or dec is None or mag is None:
            continue
        if not (ra_min <= ra <= ra_max and dec_min <= dec <= dec_max):
            continue  # cone search returns a circle - clip down to the actual box
        source_id_raw = row.get("source_id")
        out.append((
            # int() directly, not int(float(...)) - Gaia source_ids are ~19
            # digits, well past a double's exact-integer range (2^53), so
            # routing through float() first would silently round them.
            int(source_id_raw) if source_id_raw else None,
            ra, dec, mag,
            _to_float(row.get("pmra")), _to_float(row.get("pmdec")),
            _to_float(row.get("teff_gspphot")), _to_float(row.get("bp_rp")), _to_float(row.get("parallax")),
        ))
    out.sort(key=lambda r: r[3])
    return out[:max_stars]


def _query_gaia_tap(adql):
    resp = requests.get(
        GAIA_TAP_URL,
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql},
        # A plain ra/dec BETWEEN scan over gaiadr3.gaia_source (billions of
        # rows) isn't spatially indexed the way a cone search would be, so
        # the sync TAP endpoint can legitimately take a while under load -
        # matches the 90s timeout plate_solve.py already uses for its
        # (larger) bulk query against the same table.
        timeout=60,
    )
    resp.raise_for_status()
    return [tuple(row) for row in resp.json().get("data", [])]


def _query_gaia_stars(ra_min, ra_max, dec_min, dec_max, center_ra, center_dec, adql, max_stars):
    try:
        return _query_gaia_cone(ra_min, ra_max, dec_min, dec_max, center_ra, center_dec, max_stars)
    except Exception:  # noqa: BLE001 - any cone-search failure (network, bad VOTable, ...) falls back to TAP
        return _query_gaia_tap(adql)


@app.route("/api/stars")
def api_stars():
    state = _get_state(request.args.get("image", ""))
    missing = _require_lowres(state)
    if missing:
        return missing

    try:
        x0, y0, x1, y1 = _parse_box(request.args, state.lowres_width, state.lowres_height)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        max_stars = int(request.args.get("max_stars", DEFAULT_MAX_STARS))
    except ValueError:
        max_stars = DEFAULT_MAX_STARS
    max_stars = max(1, min(max_stars, MAX_STARS_LIMIT))

    ra0_rad, dec0_rad, plate_cx, plate_cy = _plate_params(state)
    plate_arcsec_per_px = 206264.80624709636 / math.hypot(plate_cx[1], plate_cx[2])

    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    radec_corners = [lowres_pixel_to_radec(px, py, ra0_rad, dec0_rad, plate_cx, plate_cy) for px, py in corners]
    ras = [c[0] for c in radec_corners]
    decs = [c[1] for c in radec_corners]
    ra_min, ra_max = min(ras), max(ras)
    dec_min, dec_max = min(decs), max(decs)
    center_ra, center_dec = lowres_pixel_to_radec(
        (x0 + x1) / 2.0, (y0 + y1) / 2.0, ra0_rad, dec0_rad, plate_cx, plate_cy
    )

    sky_region = {
        "ra_min": ra_min, "ra_max": ra_max,
        "dec_min": dec_min, "dec_max": dec_max,
        "center_ra": center_ra, "center_dec": center_dec,
        "center_ra_str": format_ra(center_ra), "center_dec_str": format_dec(center_dec),
        "ra_min_str": format_ra(ra_min), "ra_max_str": format_ra(ra_max),
        "dec_min_str": format_dec(dec_min), "dec_max_str": format_dec(dec_max),
        "width_arcmin": (x1 - x0) * plate_arcsec_per_px / 60.0,
        "height_arcmin": (y1 - y0) * plate_arcsec_per_px / 60.0,
        "lowres_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }

    if state.hires_available:
        scale_x = state.hires_width / state.lowres_width
        scale_y = state.hires_height / state.lowres_height
        # Mirror the int-floored bounds crop_hires actually extracts, so marker
        # fractions computed against this box line up with the delivered pixels.
        hx0 = max(0, min(int(x0 * scale_x), state.hires_width - 1))
        hy0 = max(0, min(int(y0 * scale_y), state.hires_height - 1))
        hx1 = max(hx0 + 1, min(int(x1 * scale_x), state.hires_width))
        hy1 = max(hy0 + 1, min(int(y1 * scale_y), state.hires_height))
        sky_region["hires_box"] = {"x0": hx0, "y0": hy0, "x1": hx1, "y1": hy1}

    adql = (
        f"SELECT TOP {max_stars} source_id, ra, dec, phot_g_mean_mag, pmra, pmdec, teff_gspphot, bp_rp, parallax "
        "FROM gaiadr3.gaia_source "
        f"WHERE ra BETWEEN {ra_min!r} AND {ra_max!r} "
        f"AND dec BETWEEN {dec_min!r} AND {dec_max!r} "
        "AND phot_g_mean_mag IS NOT NULL "
        "ORDER BY phot_g_mean_mag ASC"
    )

    stars = []
    query_error = None
    try:
        rows = _query_gaia_stars(ra_min, ra_max, dec_min, dec_max, center_ra, center_dec, adql, max_stars)
        for row in rows:
            source_id, ra, dec, mag, pmra, pmdec, teff, bp_rp, parallax = row
            # Gaia DR3 positions are for epoch J2016.0; propagate proper motion
            # forward to the image's observation epoch before locating the
            # star in the image (up to ~0.5 arcsec for fast movers).
            ra_img, dec_img = ra, dec
            if pmra is not None and pmdec is not None:
                dt = state.cfg.image_epoch - GAIA_EPOCH
                dec_img = dec + (pmdec * dt / 3.6e6)
                ra_img = ra + (pmra * dt / 3.6e6) / math.cos(math.radians(dec))
            lx, ly = radec_to_lowres_pixel(ra_img, dec_img, ra0_rad, dec0_rad, plate_cx, plate_cy)
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
            if state.hires_available:
                # Pixel centers map between resolutions as (i + 0.5) * scale, not
                # i * scale - the naive form is off by 0.5*scale - 0.5 px, a
                # constant shift that's invisible at lowres but obvious when a
                # small selection is zoomed.
                star["hires_x"] = (lx + 0.5) * (state.hires_width / state.lowres_width)
                star["hires_y"] = (ly + 0.5) * (state.hires_height / state.lowres_height)
            stars.append(star)
    except requests.RequestException as exc:
        query_error = f"Could not reach the Gaia archive: {exc}"
    except (ValueError, KeyError) as exc:
        query_error = f"Unexpected response from the Gaia archive: {exc}"

    return jsonify({"sky_region": sky_region, "stars": stars, "query_error": query_error})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

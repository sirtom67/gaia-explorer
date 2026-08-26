"""Plate-solve a registered image's preview JPEG against real Gaia DR3 positions.

Produces the plate_cx / plate_cy polynomial coefficients to paste into the
image's ImageConfig in images.py.

History: the first version of this script used ~150 stars (G<13) and fit a
rigid (rotation+scale+translation) transform. That gave a good *average* fit
(median ~1px residual) but was locally biased by up to ~20px in hires pixels
(~4 arcsec) in under-sampled parts of the frame, since a 4-parameter rigid
transform fit from too few, sparse points can't average out local noise -
users noticed markers landing visibly off-star in some crops. This version
uses thousands of stars spanning the whole frame and fits a degree-3
polynomial distortion on top of the gnomonic/TAN projection (similar to a
SIP-distorted WCS), cutting the residual to median ~0.4px.

Approach:
1. Query all Gaia stars brighter than --mag-limit across the field - dense,
   full coverage - and propagate them from Gaia's J2016 epoch to the image's
   observation epoch, the same way app.py does when placing markers.
2. Detect point-source peaks across the whole lowres JPEG (brightness
   threshold + connected-component labeling, peak pixel per blob).
3. Match each Gaia star to the nearest detected peak, then fit, in several
   passes with shrinking match radii (see _STAGES). Starting loose matters
   when seeding from the linear fallback, whose systematic error can exceed
   a tight match radius; each pass re-matches against the improved fit.
4. Re-fit with iterative outlier rejection.

Note the fit targets each peak's integer pixel index, while app.py plots
markers at index + 0.5 (the pixel's center). That +0.5 is deliberate and the
two halves cancel: a star whose brightest pixel is index i has its true
centroid at i + 0.5 on average.

Usage: python plate_solve.py [--image KEY] [--mag-limit MAG]
  KEY is one of the keys in images.IMAGES (default: ocean_of_stars). The
  target image's low-res JPEG must already be downloaded (via the app's UI,
  or manually to flask_app/static/img/<filename>).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from images import IMAGES, linear_plate_solution  # noqa: E402

# ARI Heidelberg's Gaia TAP mirror. Used instead of ESA's own endpoint
# because some networks can reach the rest of gea.esac.esa.int but hang on
# its TAP sync path (see the GAIA_CONE_URL note in app.py). Queries here
# must use the indexed CONTAINS/CIRCLE form - a plain ra/dec BETWEEN scan
# over gaiadr3.gaia_source is unindexed and times out.
GAIA_TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap/sync"
GAIA_EPOCH = 2016.0
DEFAULT_MAG_LIMIT = 16.0
DETECT_THRESHOLD = 200
MAX_BLOB_PX = 300
POLY_DEGREE = 3
MAXREC = 400000

# (match radius in px, polynomial degree) per pass. The first pass is
# deliberately loose and low-order: it only needs to pull a linear-fallback
# seed into place, and a high-order fit on loosely-matched pairs would bend
# toward the mismatches.
_STAGES = ((40, 1), (15, 3), (6, 3), (4, 3))
_NCOEF = {1: 3, 2: 6, 3: 10}


def gnomonic(ra_deg, dec_deg, ra0, dec0):
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    dra = ra - ra0
    denom = math.sin(dec0) * math.sin(dec) + math.cos(dec0) * math.cos(dec) * math.cos(dra)
    xi = math.cos(dec) * math.sin(dra) / denom
    eta = (math.cos(dec0) * math.sin(dec) - math.sin(dec0) * math.cos(dec) * math.cos(dra)) / denom
    return xi, eta


def design_row(xi, eta, degree):
    terms = [1.0, xi, eta]
    if degree >= 2:
        terms += [xi * xi, xi * eta, eta * eta]
    if degree >= 3:
        terms += [xi ** 3, xi * xi * eta, xi * eta * eta, eta ** 3]
    return terms


def predict(xi, eta, cx, cy, degree):
    row = design_row(xi, eta, degree)
    n = _NCOEF[degree]
    return (sum(c * r for c, r in zip(cx[:n], row)),
            sum(c * r for c, r in zip(cy[:n], row)))


def fit(matches, degree):
    xis = np.array([m["xi"] for m in matches])
    etas = np.array([m["eta"] for m in matches])
    oxs = np.array([m["obs_x"] for m in matches])
    oys = np.array([m["obs_y"] for m in matches])
    basis = np.array([design_row(x, e, degree) for x, e in zip(xis, etas)])
    ncoef = basis.shape[1]
    n = len(xis)
    A = np.zeros((2 * n, 2 * ncoef))
    b_vec = np.zeros(2 * n)
    A[0:n, 0:ncoef] = basis
    A[n:, ncoef:] = basis
    b_vec[0:n] = oxs
    b_vec[n:] = oys
    sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    cx, cy = sol[0:ncoef], sol[ncoef:]
    resid = np.hypot(basis @ cx - oxs, basis @ cy - oys)
    return cx, cy, resid


def detect_peaks(img):
    """Brightest pixel of each small bright blob - i.e. candidate point sources."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    mask = arr > DETECT_THRESHOLD
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    valid = [i + 1 for i, s in enumerate(sizes) if 1 <= s <= MAX_BLOB_PX]
    maxpos = ndimage.maximum_position(arr, labeled, valid)
    return np.array([(p[1], p[0]) for p in maxpos], dtype=float)


def query_gaia(cfg, mag_limit):
    """All Gaia sources brighter than mag_limit within the frame's circumscribed
    circle, propagated to the image epoch and projected to gnomonic xi/eta."""
    half_diag_deg = math.hypot(*cfg.field_of_view_arcmin) / 60.0 / 2.0
    radius = half_diag_deg * 1.05
    adql = (
        "SELECT source_id, ra, dec, phot_g_mean_mag, pmra, pmdec FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
        f"CIRCLE('ICRS',{cfg.center_ra_deg},{cfg.center_dec_deg},{radius})) "
        f"AND phot_g_mean_mag < {mag_limit}"
    )
    resp = requests.get(GAIA_TAP_URL, params={
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
        "QUERY": adql, "MAXREC": MAXREC,
    }, timeout=600)
    resp.raise_for_status()
    rows = resp.json()["data"]
    if len(rows) >= MAXREC:
        print(f"WARNING: hit MAXREC ({MAXREC}) - the field is only partly covered; "
              f"lower --mag-limit for an unbiased fit.")
    print(f"Gaia returned {len(rows)} stars (G < {mag_limit})")

    cols = ("source_id", "ra", "dec", "phot_g_mean_mag", "pmra", "pmdec")
    ra0, dec0 = math.radians(cfg.center_ra_deg), math.radians(cfg.center_dec_deg)
    dt = cfg.image_epoch - GAIA_EPOCH
    stars = []
    for row in rows:
        d = row if isinstance(row, dict) else dict(zip(cols, row))
        ra, dec = d["ra"], d["dec"]
        pmra, pmdec = d.get("pmra"), d.get("pmdec")
        if pmra is not None and pmdec is not None:
            dec_img = dec + pmdec * dt / 3.6e6
            ra_img = ra + (pmra * dt / 3.6e6) / math.cos(math.radians(dec))
            ra, dec = ra_img, dec_img
        xi, eta = gnomonic(ra, dec, ra0, dec0)
        stars.append({"xi": xi, "eta": eta})
    return stars


def match(stars, peaks, cx, cy, degree, radius, width, height):
    out = []
    for s in stars:
        px, py = predict(s["xi"], s["eta"], cx, cy, degree)
        if not (0 <= px < width and 0 <= py < height):
            continue
        d = np.hypot(peaks[:, 0] - px, peaks[:, 1] - py)
        j = int(np.argmin(d))
        if d[j] <= radius:
            m = dict(s)
            m["obs_x"], m["obs_y"] = peaks[j]
            out.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="ocean_of_stars", choices=sorted(IMAGES),
                        help="image key from images.py (default: ocean_of_stars)")
    parser.add_argument("--mag-limit", type=float, default=DEFAULT_MAG_LIMIT,
                        help=f"Gaia G magnitude cut (default: {DEFAULT_MAG_LIMIT})")
    args = parser.parse_args()
    cfg = IMAGES[args.image]

    if not cfg.lowres_path.exists():
        sys.exit(f"{cfg.lowres_path} doesn't exist yet - download it first (via the app's UI, or manually).")

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(cfg.lowres_path)
    width, height = img.size
    print(f"{args.image}: {width} x {height} preview")

    peaks = detect_peaks(img)
    print(f"{len(peaks)} point-source peaks detected (threshold={DETECT_THRESHOLD})")

    stars = query_gaia(cfg, args.mag_limit)

    # Seed from the image's existing fit if it has one, else the linear fallback.
    if cfg.plate_cx is not None:
        cx, cy, degree = cfg.plate_cx, cfg.plate_cy, POLY_DEGREE
        print("seeding from this image's existing plate solution")
    else:
        cx, cy = linear_plate_solution(width, height, cfg.field_of_view_arcmin)
        degree = 1
        print("seeding from the linear (undistorted) fallback")

    matches = []
    for radius, stage_degree in _STAGES:
        matches = match(stars, peaks, cx, cy, degree, radius, width, height)
        if len(matches) < _NCOEF[stage_degree] * 5:
            sys.exit(f"only {len(matches)} matches within {radius}px - too few to fit. "
                     f"Try a fainter --mag-limit, or check the image's center_ra/dec "
                     f"and field_of_view_arcmin in images.py.")
        cx, cy, resid = fit(matches, stage_degree)
        degree = stage_degree
        print(f"  radius {radius:>3}px degree {stage_degree}: n={len(matches):<6} "
              f"resid mean={resid.mean():.3f} median={np.median(resid):.3f} max={resid.max():.3f}")

    for _ in range(3):
        cx, cy, resid = fit(matches, POLY_DEGREE)
        keep = resid <= max(2.0, np.median(resid) * 4)
        if keep.all():
            break
        matches = [m for m, k in zip(matches, keep) if k]
        cx, cy, resid = fit(matches, POLY_DEGREE)
        print(f"  outlier reject: n={len(matches):<6} "
              f"resid mean={resid.mean():.3f} median={np.median(resid):.3f} max={resid.max():.3f}")

    print()
    print(f"Paste into images.py IMAGES[{args.image!r}]:")
    print(f"        plate_cx={tuple(cx)!r},")
    print(f"        plate_cy={tuple(cy)!r},")
    print(f"scale arcsec/px = {206264.80624709636 / math.hypot(cx[1], cx[2])!r}")


if __name__ == "__main__":
    main()

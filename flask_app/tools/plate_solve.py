"""Plate-solve a registered image's preview JPEG against real Gaia DR3 positions.

Produces the plate_cx / plate_cy polynomial coefficients to paste into the
image's ImageConfig in images.py.

History: an earlier version of this script used ~150 stars (G<13) and fit a
rigid (rotation+scale+translation) transform. That gave a good *average* fit
(median ~1px residual) but was locally biased by up to ~20px in hires pixels
(~4 arcsec) in under-sampled parts of the frame, since a 4-parameter rigid
transform fit from too few, sparse points can't average out local noise -
users noticed markers landing visibly off-star in some crops. This version
uses ~4500 stars (G<14.5) spanning the whole frame and fits a degree-3
polynomial distortion on top of the gnomonic/TAN projection (similar to a
SIP-distorted WCS), cutting the residual to median ~0.4px, max ~2.5px
everywhere tested (for Ocean of Stars; results will vary per image).

Approach:
1. Query all Gaia stars (G < 14.5) across the field - dense, full coverage.
2. Detect point-source peaks across the whole lowres JPEG (brightness
   threshold + connected-component labeling, peak pixel per blob).
3. Match each Gaia star to the nearest detected peak using a seed plate
   solution (the image's hardcoded fit if it has one, otherwise the linear
   fallback), with a tight match radius.
4. Fit a degree-3 polynomial (in gnomonic xi/eta) to observed pixel
   positions via linear least squares, with iterative outlier rejection.

Usage: python plate_solve.py [--image KEY]
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

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
MAG_LIMIT = 14.5
DETECT_THRESHOLD = 200
MATCH_RADIUS_PX = 8
POLY_DEGREE = 3


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


def seed_predict(ra, dec, ra0, dec0, seed_cx, seed_cy):
    xi, eta = gnomonic(ra, dec, ra0, dec0)
    row = design_row(xi, eta, POLY_DEGREE)
    return sum(c * r for c, r in zip(seed_cx, row)), sum(c * r for c, r in zip(seed_cy, row))


def fit(matches, degree, ra0, dec0):
    xis, etas, oxs, oys = [], [], [], []
    for m in matches:
        xi, eta = gnomonic(m["ra"], m["dec"], ra0, dec0)
        xis.append(xi); etas.append(eta); oxs.append(m["obs_x"]); oys.append(m["obs_y"])
    xis, etas, oxs, oys = map(np.array, (xis, etas, oxs, oys))
    rows = [design_row(x, e, degree) for x, e in zip(xis, etas)]
    basis = np.array(rows)
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
    pred_x = basis @ cx
    pred_y = basis @ cy
    resid = np.hypot(pred_x - oxs, pred_y - oys)
    return cx, cy, resid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="ocean_of_stars", choices=sorted(IMAGES),
                         help="image key from images.py (default: ocean_of_stars)")
    args = parser.parse_args()
    cfg = IMAGES[args.image]

    if not cfg.lowres_path.exists():
        sys.exit(f"{cfg.lowres_path} doesn't exist yet - download it first (via the app's UI, or manually).")

    ra0_deg, dec0_deg = cfg.center_ra_deg, cfg.center_dec_deg
    ra0, dec0 = math.radians(ra0_deg), math.radians(dec0_deg)

    img = Image.open(cfg.lowres_path)
    master_w, master_h = img.size

    if cfg.plate_cx is not None:
        seed_cx, seed_cy = cfg.plate_cx, cfg.plate_cy
    else:
        seed_cx, seed_cy = linear_plate_solution(master_w, master_h, cfg.field_of_view_arcmin)

    half_w_deg = cfg.field_of_view_arcmin[0] / 60.0 / 2.0 + 0.1
    half_h_deg = cfg.field_of_view_arcmin[1] / 60.0 / 2.0 + 0.1
    adql = (
        f"SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
        f"WHERE ra BETWEEN {ra0_deg - half_w_deg} AND {ra0_deg + half_w_deg} "
        f"AND dec BETWEEN {dec0_deg - half_h_deg} AND {dec0_deg + half_h_deg} "
        f"AND phot_g_mean_mag < {MAG_LIMIT} ORDER BY phot_g_mean_mag ASC"
    )
    resp = requests.get(GAIA_TAP_URL, params={
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql,
    }, timeout=90)
    resp.raise_for_status()
    rows = resp.json()["data"]
    print(f"Gaia returned {len(rows)} candidates (G < {MAG_LIMIT})")

    arr = np.asarray(img.convert("L"), dtype=np.float32)
    mask = arr > DETECT_THRESHOLD
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes) if 1 <= s <= 300]
    maxpos = ndimage.maximum_position(arr, labeled, valid_labels)
    peak_xy = np.array([(p[1], p[0]) for p in maxpos])
    print(f"{len(peak_xy)} point-source peaks detected (threshold={DETECT_THRESHOLD})")

    matches = []
    for source_id, ra, dec, mag in rows:
        pred_x, pred_y = seed_predict(ra, dec, ra0, dec0, seed_cx, seed_cy)
        if not (0 <= pred_x < master_w and 0 <= pred_y < master_h):
            continue
        d = np.hypot(peak_xy[:, 0] - pred_x, peak_xy[:, 1] - pred_y)
        j = int(np.argmin(d))
        if d[j] <= MATCH_RADIUS_PX:
            matches.append(dict(source_id=source_id, ra=ra, dec=dec, mag=mag,
                                 obs_x=peak_xy[j, 0], obs_y=peak_xy[j, 1]))
    print(f"{len(matches)} matched within {MATCH_RADIUS_PX}px of seed prediction")

    cur = matches
    cx, cy, resid = fit(cur, POLY_DEGREE, ra0, dec0)
    print(f"Round 1: n={len(cur)} mean={resid.mean():.3f} median={np.median(resid):.3f} max={resid.max():.3f}")
    for _ in range(3):
        keep = resid <= max(2.5, np.median(resid) * 4)
        if keep.all():
            break
        cur = [m for m, k in zip(cur, keep) if k]
        cx, cy, resid = fit(cur, POLY_DEGREE, ra0, dec0)
        print(f"Round: n={len(cur)} mean={resid.mean():.3f} median={np.median(resid):.3f} max={resid.max():.3f}")

    print()
    print(f"Paste into images.py IMAGES[{args.image!r}]:")
    print(f"plate_cx = {tuple(cx)!r}")
    print(f"plate_cy = {tuple(cy)!r}")
    print(f"scale arcsec/px = {206264.80624709636 / math.hypot(cx[1], cx[2])!r}")


if __name__ == "__main__":
    main()

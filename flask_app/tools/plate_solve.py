"""Plate-solve the Ocean of Stars preview JPEG against real Gaia DR3 positions.

Produces the PLATE_CX / PLATE_CY polynomial coefficients hardcoded in app.py.
Re-run this if the preview JPEG is ever regenerated at a different size/crop.

History: an earlier version of this script used ~150 stars (G<13) and fit a
rigid (rotation+scale+translation) transform. That gave a good *average* fit
(median ~1px residual) but was locally biased by up to ~20px in hires pixels
(~4 arcsec) in under-sampled parts of the frame, since a 4-parameter rigid
transform fit from too few, sparse points can't average out local noise -
users noticed markers landing visibly off-star in some crops. This version
uses ~4500 stars (G<14.5) spanning the whole frame and fits a degree-3
polynomial distortion on top of the gnomonic/TAN projection (similar to a
SIP-distorted WCS), cutting the residual to median ~0.4px, max ~2.5px
everywhere tested, including the previously-bad region.

Approach:
1. Query all Gaia stars (G < 14.5) across the field - dense, full coverage.
2. Detect point-source peaks across the whole lowres JPEG (brightness
   threshold + connected-component labeling, peak pixel per blob).
3. Match each Gaia star to the nearest detected peak using the *current*
   plate solution (from app.py) as the seed, with a tight match radius.
4. Fit a degree-3 polynomial (in gnomonic xi/eta) to observed pixel
   positions via linear least squares, with iterative outlier rejection.

Usage: python plate_solve.py
"""
import math

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

MASTER_JPG = "../static/img/ocean_of_stars.jpg"

RA0_DEG = 224.76984640789192
DEC0_DEG = -37.939446521397656
RA0 = math.radians(RA0_DEG)
DEC0 = math.radians(DEC0_DEG)

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
MAG_LIMIT = 14.5
DETECT_THRESHOLD = 200
MATCH_RADIUS_PX = 8
POLY_DEGREE = 3

# Current plate solution (from app.py), used only to seed the search.
_SEED_CX = (1999.7636315234488, -73088.16239545682, 97.47720777489019,
            -2427.235450950923, -4.351627915670664, -169.98195207141725,
            86835.41713612381, 1921.6427119784203, 676.4196286585695, -18562.04584138326)
_SEED_CY = (1060.9556173595986, -98.86461543816775, -73115.4770637289,
            -1220.0426883027787, -885.956838531343, 18.552137270575322,
            -8218.147610689712, 102.28652799927075, 1529.3802893802067, 8569.652326636984)


def gnomonic(ra_deg, dec_deg):
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    dra = ra - RA0
    denom = math.sin(DEC0) * math.sin(dec) + math.cos(DEC0) * math.cos(dec) * math.cos(dra)
    xi = math.cos(dec) * math.sin(dra) / denom
    eta = (math.cos(DEC0) * math.sin(dec) - math.sin(DEC0) * math.cos(dec) * math.cos(dra)) / denom
    return xi, eta


def design_row(xi, eta, degree):
    terms = [1.0, xi, eta]
    if degree >= 2:
        terms += [xi * xi, xi * eta, eta * eta]
    if degree >= 3:
        terms += [xi ** 3, xi * xi * eta, xi * eta * eta, eta ** 3]
    return terms


def seed_predict(ra, dec):
    xi, eta = gnomonic(ra, dec)
    row = design_row(xi, eta, POLY_DEGREE)
    return sum(c * r for c, r in zip(_SEED_CX, row)), sum(c * r for c, r in zip(_SEED_CY, row))


def fit(matches, degree):
    xis, etas, oxs, oys = [], [], [], []
    for m in matches:
        xi, eta = gnomonic(m["ra"], m["dec"])
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
    img = Image.open(MASTER_JPG)
    master_w, master_h = img.size

    half_w_deg = 2.05
    half_h_deg = 0.95
    adql = (
        f"SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
        f"WHERE ra BETWEEN {RA0_DEG - half_w_deg} AND {RA0_DEG + half_w_deg} "
        f"AND dec BETWEEN {DEC0_DEG - half_h_deg} AND {DEC0_DEG + half_h_deg} "
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
        pred_x, pred_y = seed_predict(ra, dec)
        if not (0 <= pred_x < master_w and 0 <= pred_y < master_h):
            continue
        d = np.hypot(peak_xy[:, 0] - pred_x, peak_xy[:, 1] - pred_y)
        j = int(np.argmin(d))
        if d[j] <= MATCH_RADIUS_PX:
            matches.append(dict(source_id=source_id, ra=ra, dec=dec, mag=mag,
                                 obs_x=peak_xy[j, 0], obs_y=peak_xy[j, 1]))
    print(f"{len(matches)} matched within {MATCH_RADIUS_PX}px of seed prediction")

    cur = matches
    cx, cy, resid = fit(cur, POLY_DEGREE)
    print(f"Round 1: n={len(cur)} mean={resid.mean():.3f} median={np.median(resid):.3f} max={resid.max():.3f}")
    for _ in range(3):
        keep = resid <= max(2.5, np.median(resid) * 4)
        if keep.all():
            break
        cur = [m for m, k in zip(cur, keep) if k]
        cx, cy, resid = fit(cur, POLY_DEGREE)
        print(f"Round: n={len(cur)} mean={resid.mean():.3f} median={np.median(resid):.3f} max={resid.max():.3f}")

    print()
    print(f"PLATE_CX = {tuple(cx)!r}")
    print(f"PLATE_CY = {tuple(cy)!r}")
    print(f"scale arcsec/px = {206264.80624709636 / math.hypot(cx[1], cx[2])!r}")


if __name__ == "__main__":
    main()

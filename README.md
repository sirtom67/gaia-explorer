# Gaia Explorer

Tools for exploring gigapixel NSF–DOE Vera C. Rubin Observatory images.

## flask_app — Rubin Observatory Subimage Explorer

A local Flask web app for panning around a full-frame Rubin image and pulling
out subimages at full resolution. Two images are currently registered (see
`flask_app/images.py`):

- **[Ocean of Stars](https://noirlab.edu/public/images/noirlab2616a/)**
  (`noirlab2616a`) — a 1.7-gigapixel, 56428 × 29949 px view of a star field
  in Lupus, taken with the LSST Camera at the start of the Legacy Survey of
  Space and Time.
- **[Cosmos](https://noirlab.edu/public/images/noirlab2618a/)** (`noirlab2618a`)
  — a 55536 × 30291 px deep coadd of the COSMOS field in Sextans, released
  for Rubin's Early Data Preview 2.

More images can be added by appending an `ImageConfig` entry to
`flask_app/images.py`.

- **Pick an image** from the picker at the top of the page. If its low-res
  preview JPEG hasn't been downloaded yet, a "Download preview" button fetches
  it from NOIRLab directly into the app (a few MB, seconds).
- **Drag-select a region** on the full-frame preview map and extract the
  matching crop in two flavors: the low-res preview JPEG ("Low res" tab) and
  the full-resolution 16-bit TIFF ("High res" tab). If the full-res TIFF
  (~9.4 GB) hasn't been downloaded yet, the "High res" tab shows a download
  button instead — click it to fetch it in the background, with a progress
  bar. The TIFF is read via a numpy memmap, so crops are fast and the file is
  never loaded into RAM.
- **Zoom by dragging on an extracted image** — the dragged rectangle becomes
  the new selection and re-extracts automatically, so you can iteratively
  dive down to native TIFF resolution.
- **Live Gaia DR3 star data**: each extraction queries the ESA Gaia archive
  for the 10 brightest stars in the selected sky area, showing designation,
  G magnitude, estimated spectral type, distance (from parallax, in light
  years), and proper motion. Click a star to highlight it on the crops and
  draw its +100-year proper-motion vector; a scale bar shows the angular
  scale.
- Sky coordinates come from a plate solution: a real fit against thousands of
  detected Gaia star positions where available (gnomonic projection +
  degree-3 polynomial distortion — see `flask_app/tools/plate_solve.py`), or
  a linear (undistorted) TAN projection derived from the image's published
  field of view otherwise, with star positions propagated from Gaia's J2016
  epoch to each image's approximate observation epoch.

  Both bundled images have a real plate solve (median residual ~0.4 preview
  px, measured against flux-weighted stellar centroids). The linear fallback
  only applies to a newly added image that hasn't been solved yet — it is
  good to roughly a pixel, which is invisible at full-frame zoom but throws
  markers visibly off-star once a small selection is blown up. To solve a new
  image, download its preview JPEG and run:

  ```
  python flask_app/tools/plate_solve.py --image <key>
  ```

  then paste the printed `plate_cx`/`plate_cy` tuples into its `ImageConfig`
  in `flask_app/images.py`.

### Setup

1. Install dependencies (Python 3.10+):

   ```
   pip install -r flask_app/requirements.txt
   ```

2. Run the app and open http://localhost:5000:

   ```
   python flask_app/app.py
   ```

   Ocean of Stars' preview JPEG is already bundled at
   `flask_app/static/img/ocean_of_stars.jpg`. Every other image/file —
   Cosmos's preview JPEG, and both images' full-resolution TIFFs — is fetched
   on demand from the app's UI the first time you select it, and cached
   locally (`flask_app/static/img/` for previews, `flask_app/data/` for
   TIFFs) so it's only downloaded once.

   To point a full-res TIFF at a file you already downloaded elsewhere
   instead, set its environment variable before starting the app:

   ```
   set OCEAN_OF_STARS_TIFF=D:\path\to\noirlab2616a.tif
   set COSMOS_TIFF=D:\path\to\noirlab2618a.tif
   ```

The Gaia star table needs internet access (it queries
`gea.esac.esa.int` live), as does any in-app download; everything else is
local.

### Credits

Images: NSF–DOE Vera C. Rubin Observatory / NOIRLab / SLAC / AURA
([usage terms](https://noirlab.edu/public/copyright/)).
Star data: ESA [Gaia](https://www.cosmos.esa.int/gaia) DR3.

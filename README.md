# Gaia Explorer

Tools for exploring the NSF–DOE Vera C. Rubin Observatory's **"Ocean of Stars"**
image ([noirlab2616a](https://noirlab.edu/public/images/noirlab2616a/)) — a
1.7-gigapixel, 56428 × 29949 px view of a star field in Lupus, taken with the
LSST Camera at the start of the Legacy Survey of Space and Time.

## flask_app — Ocean of Stars Subimage Explorer

A local Flask web app for panning around the full image and pulling out
subimages at full resolution:

- **Drag-select a region** on the full-frame preview map and extract the
  matching crop in two flavors: the bundled 4000 × 2123 preview JPEG
  ("Low res" tab) and the full-resolution 16-bit TIFF ("High res" tab).
  The TIFF is read via a numpy memmap, so crops are fast and the 10 GB file
  is never loaded into RAM.
- **Zoom by dragging on an extracted image** — the dragged rectangle becomes
  the new selection and re-extracts automatically, so you can iteratively
  dive down to native TIFF resolution.
- **Live Gaia DR3 star data**: each extraction queries the ESA Gaia archive
  for the 10 brightest stars in the selected sky area, showing designation,
  G magnitude, estimated spectral type, distance (from parallax, in light
  years), and proper motion. Click a star to highlight it on the crops and
  draw its +100-year proper-motion vector; a scale bar shows the angular
  scale.
- Sky coordinates come from a plate solution fit against ~4500 detected Gaia
  star positions (gnomonic projection + degree-3 polynomial distortion,
  median residual ~0.4 preview px — see `flask_app/tools/plate_solve.py`),
  with star positions propagated from Gaia's J2016 epoch to the image epoch.

### Setup

1. Install dependencies (Python 3.10+):

   ```
   pip install -r flask_app/requirements.txt
   ```

2. **Download the full-resolution TIFF** (optional but recommended — without
   it the app still runs, just with the "High res" tab disabled):

   Go to the [NOIRLab image page](https://noirlab.edu/public/images/noirlab2616a/)
   and download the **Fullsize Original** (`noirlab2616a.tif`, 9.4 GB), or grab
   it directly:

   ```
   https://storage.noirlab.edu/media/archives/images/original/noirlab2616a.tif
   ```

   Then point the app at it with the `OCEAN_OF_STARS_TIFF` environment
   variable (default: `C:\Users\thoma\Downloads\noirlab2616a.tif`):

   ```
   set OCEAN_OF_STARS_TIFF=D:\path\to\noirlab2616a.tif
   ```

   The preview JPEG (the 4000 × 2123 "Publication JPEG" from the same page)
   is already bundled at `flask_app/static/img/ocean_of_stars.jpg`.

3. Run the app and open http://localhost:5000:

   ```
   python flask_app/app.py
   ```

The Gaia star table needs internet access (it queries
`gea.esac.esa.int` live); everything else is local.

### Credits

Image: NSF–DOE Vera C. Rubin Observatory / NOIRLab / SLAC / AURA
([usage terms](https://noirlab.edu/public/copyright/)).
Star data: ESA [Gaia](https://www.cosmos.esa.int/gaia) DR3.

"""Registry and runtime state for the Rubin Observatory images this app can browse.

Each image has a small "low-res" JPEG used for the on-screen preview map, and
an optional full-resolution 16-bit TIFF used for native-detail crops. Neither
file needs to be present on disk at startup: both can be fetched on demand
(see start_download/get_download_status) and are (re)loaded automatically
once they exist.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import tifffile
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
STATIC_IMG_DIR = BASE_DIR / "static" / "img"
HIRES_DATA_DIR = BASE_DIR / "data"

# Cap on the largest dimension we'll ever materialize/send for a hi-res crop,
# so a huge drag selection can't try to read gigabytes at once.
MAX_HIRES_OUTPUT_DIM = 4000


@dataclass(frozen=True)
class ImageConfig:
    key: str
    title: str
    credit: str
    source_url: str
    center_ra_deg: float
    center_dec_deg: float
    field_of_view_arcmin: tuple  # (width, height)
    image_epoch: float  # approximate observation epoch (Julian year), for proper-motion propagation
    lowres_url: str
    lowres_filename: str
    lowres_size_hint_bytes: Optional[int]
    hires_url: str
    hires_env_var: str
    hires_filename: str
    hires_size_hint_bytes: Optional[int]
    # Degree-3 polynomial distortion coefficients from a real plate solve
    # (see tools/plate_solve.py, and app.py's PLATE_CX/CY comment for the
    # basis they're expressed in). None -> ImageState.plate_solution() falls
    # back to a plain linear TAN projection derived from field_of_view_arcmin.
    plate_cx: Optional[tuple] = None
    plate_cy: Optional[tuple] = None

    @property
    def lowres_path(self) -> Path:
        return STATIC_IMG_DIR / self.lowres_filename

    @property
    def hires_path(self) -> Path:
        return Path(os.environ.get(self.hires_env_var, str(HIRES_DATA_DIR / self.hires_filename)))


IMAGES = {
    "ocean_of_stars": ImageConfig(
        key="ocean_of_stars",
        title="Ocean of Stars",
        credit="NSF-DOE Vera C. Rubin Observatory/NOIRLab/SLAC/AURA",
        source_url="https://noirlab.edu/public/images/noirlab2616a/",
        center_ra_deg=224.76984640789192,
        center_dec_deg=-37.939446521397656,
        field_of_view_arcmin=(188.33, 99.95),
        image_epoch=2025.5,
        lowres_url="https://storage.noirlab.edu/media/archives/images/publicationjpg/noirlab2616a.jpg",
        lowres_filename="ocean_of_stars.jpg",
        lowres_size_hint_bytes=None,  # already bundled in the repo
        hires_url="https://storage.noirlab.edu/media/archives/images/original/noirlab2616a.tif",
        hires_env_var="OCEAN_OF_STARS_TIFF",
        hires_filename="noirlab2616a.tif",
        hires_size_hint_bytes=9_400_000_000,
        # Fit against ~4500 real Gaia DR3 stars - see tools/plate_solve.py.
        # Median residual ~0.4px, max ~2.5px across the whole frame.
        plate_cx=(
            1999.7636315234488, -73088.16239545682, 97.47720777489019,
            -2427.235450950923, -4.351627915670664, -169.98195207141725,
            86835.41713612381, 1921.6427119784203, 676.4196286585695, -18562.04584138326,
        ),
        plate_cy=(
            1060.9556173595986, -98.86461543816775, -73115.4770637289,
            -1220.0426883027787, -885.956838531343, 18.552137270575322,
            -8218.147610689712, 102.28652799927075, 1529.3802893802067, 8569.652326636984,
        ),
    ),
    "cosmos": ImageConfig(
        key="cosmos",
        title="Cosmos",
        credit="NSF-DOE Vera C. Rubin Observatory/NOIRLab/SLAC/AURA",
        source_url="https://noirlab.edu/public/images/noirlab2618a/",
        # Published field center: RA 10h 00m 38.71s, Dec +2 13 54.70
        center_ra_deg=150.16129166666667,
        center_dec_deg=2.2318611111111112,
        field_of_view_arcmin=(185.14, 100.97),
        image_epoch=2025.8,  # approximate - Rubin Early Data Preview 2 coadd, released 2026-07-31
        lowres_url="https://storage.noirlab.edu/media/archives/images/publicationjpg/noirlab2618a.jpg",
        lowres_filename="cosmos.jpg",
        lowres_size_hint_bytes=4_600_000,
        hires_url="https://storage.noirlab.edu/media/archives/images/original/noirlab2618a.tif",
        hires_env_var="COSMOS_TIFF",
        hires_filename="noirlab2618a.tif",
        hires_size_hint_bytes=9_400_000_000,
        # No dedicated plate solve yet - falls back to linear_plate_solution()
        # below. Once the JPEG has been downloaded, run:
        #   python flask_app/tools/plate_solve.py --image cosmos
        # and paste the printed plate_cx/plate_cy tuples in here for
        # sub-pixel-accurate star markers near the frame edges.
        plate_cx=None,
        plate_cy=None,
    ),
}


def linear_plate_solution(width_px, height_px, field_of_view_arcmin):
    """Fallback plate solution: a plain linear TAN projection (no distortion
    term), derived directly from the image's pixel size and its published
    field of view - principal point at image center, north up, east left
    (matching every Rubin/NOIRLab release image seen so far). Good to within
    a handful of pixels; see tools/plate_solve.py to fit a real degree-3
    polynomial distortion term against detected Gaia stars instead.
    """
    fov_w_rad = math.radians(field_of_view_arcmin[0] / 60.0)
    fov_h_rad = math.radians(field_of_view_arcmin[1] / 60.0)
    cx = (width_px / 2.0, -width_px / fov_w_rad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cy = (height_px / 2.0, 0.0, -height_px / fov_h_rad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return cx, cy


class ImageState:
    """Runtime (loaded) state for one ImageConfig - lazily (re)loaded as the
    backing files appear on disk, e.g. after an on-demand download completes."""

    def __init__(self, cfg: ImageConfig):
        self.cfg = cfg
        self.lowres_image = None
        self.lowres_width = None
        self.lowres_height = None
        self.hires_mmap = None
        self.hires_width = None
        self.hires_height = None
        self.reload()

    def reload(self):
        self._load_lowres()
        self._load_hires()

    def _load_lowres(self):
        path = self.cfg.lowres_path
        if not path.exists():
            self.lowres_image = None
            self.lowres_width = self.lowres_height = None
            return
        img = Image.open(path)
        img.load()
        self.lowres_image = img
        self.lowres_width, self.lowres_height = img.size

    def _load_hires(self):
        path = self.cfg.hires_path
        if not path.exists():
            self.hires_mmap = None
            self.hires_width = self.hires_height = None
            return
        try:
            mmap = tifffile.memmap(str(path))
            self.hires_mmap = mmap
            self.hires_height, self.hires_width = mmap.shape[0], mmap.shape[1]
        except Exception as exc:  # noqa: BLE001 - report and keep running without hi-res
            print(f"Could not open hi-res TIFF at {path}: {exc}")
            self.hires_mmap = None
            self.hires_width = self.hires_height = None

    @property
    def lowres_available(self):
        return self.lowres_image is not None

    @property
    def hires_available(self):
        return self.hires_mmap is not None

    def plate_solution(self):
        """Returns (plate_cx, plate_cy) - a hardcoded fit if this image has
        one, otherwise a linear fallback derived from the loaded JPEG's size."""
        if self.cfg.plate_cx is not None:
            return self.cfg.plate_cx, self.cfg.plate_cy
        return linear_plate_solution(self.lowres_width, self.lowres_height, self.cfg.field_of_view_arcmin)


IMAGE_STATES = {key: ImageState(cfg) for key, cfg in IMAGES.items()}


# --- On-demand downloads ----------------------------------------------------
# Simple background-thread downloader with a polled status dict - good enough
# for a single local user clicking a button, no job queue needed.

_download_lock = threading.Lock()
_download_state = {}  # (key, which) -> dict(status, bytes, total, error)


def _dest_path(cfg: ImageConfig, which: str) -> Path:
    return cfg.lowres_path if which == "lowres" else cfg.hires_path


def _source_url(cfg: ImageConfig, which: str) -> str:
    return cfg.lowres_url if which == "lowres" else cfg.hires_url


def get_download_status(key: str, which: str) -> dict:
    with _download_lock:
        return dict(_download_state.get((key, which), {"status": "idle", "bytes": 0, "total": 0, "error": None}))


def start_download(key: str, which: str):
    """Kick off a background download of the given image's file if one isn't
    already in flight. Returns (started: bool, message: str)."""
    if key not in IMAGES or which not in ("lowres", "hires"):
        return False, "unknown image or file"

    state_key = (key, which)
    with _download_lock:
        current = _download_state.get(state_key)
        if current and current["status"] == "downloading":
            return False, "already downloading"
        _download_state[state_key] = {"status": "downloading", "bytes": 0, "total": 0, "error": None}

    thread = threading.Thread(target=_download_worker, args=(key, which), daemon=True)
    thread.start()
    return True, "started"


def _download_worker(key: str, which: str):
    cfg = IMAGES[key]
    state_key = (key, which)
    dest = _dest_path(cfg, which)
    tmp = dest.with_name(dest.name + ".part")
    downloaded = 0
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(_source_url(cfg, which), stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            with _download_lock:
                _download_state[state_key] = {"status": "downloading", "bytes": 0, "total": total, "error": None}
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    with _download_lock:
                        _download_state[state_key]["bytes"] = downloaded
        tmp.replace(dest)
        with _download_lock:
            _download_state[state_key] = {"status": "done", "bytes": downloaded, "total": total, "error": None}
        IMAGE_STATES[key].reload()
    except Exception as exc:  # noqa: BLE001 - surface the error to the polling client
        tmp.unlink(missing_ok=True)
        with _download_lock:
            _download_state[state_key] = {"status": "error", "bytes": 0, "total": 0, "error": str(exc)}

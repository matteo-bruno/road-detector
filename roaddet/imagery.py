"""Aerial / satellite imagery access through XYZ tile services, with a disk cache."""
from __future__ import annotations

import io
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .geo import TILE, BBox, RasterGrid, lonlat_to_gpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileSource:
    name: str
    url: str  # template with {z} {x} {y}
    zoom: int
    attribution: str
    ext: str = "jpg"


SOURCES = {
    # Global coverage, roughly 0.3 m imagery in large cities.
    "esri": TileSource(
        "esri",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        19,
        "Esri World Imagery (Esri, Maxar, Earthstar Geographics and the GIS User Community)",
    ),
    # Esri Wayback archive: a fixed historical release of World Imagery. Release 48624
    # (2021-11-30) has diffuse light and almost no cast shadows over Rome.
    "esri_wayback": TileSource(
        "esri_wayback_48624",
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
        "default028mm/MapServer/tile/48624/{z}/{y}/{x}",
        19,
        "Esri World Imagery Wayback 2021-11-30 (Esri, Maxar, Earthstar Geographics)",
    ),
    # Netherlands, open data (CC BY 4.0), 8 cm true-ortho aerial photography.
    "pdok": TileSource(
        "pdok",
        "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg",
        21,
        "PDOK Luchtfoto Actueel HR (CC BY 4.0, beeldmateriaal.nl)",
    ),
}

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "roaddet/0.1 (research road measurement tool)"
    return _local.s


def _fetch_tile(src: TileSource, z: int, x: int, y: int, cache: Path) -> Path | None:
    path = cache / src.name / str(z) / str(x) / f"{y}.{src.ext}"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = src.url.format(z=z, x=x, y=y)
    for attempt in range(4):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 200 and r.content:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(path)
                return path
            if r.status_code == 404:
                return None
        except requests.RequestException as exc:  # network hiccup, retry
            log.debug("tile %s failed: %s", url, exc)
        time.sleep(1.5 * (attempt + 1))
    log.warning("giving up on tile %s", url)
    return None


def load_mosaic(src: TileSource, bbox: BBox, cache: Path, zoom: int | None = None,
                workers: int = 8) -> tuple[np.ndarray, RasterGrid, float]:
    """Download (or read from cache) all tiles covering bbox.

    Returns (rgb uint8 array HxWx3, grid, fraction of missing tiles).
    """
    z = zoom or src.zoom
    (xa, xb), (ya, yb) = lonlat_to_gpx([bbox.west, bbox.east], [bbox.north, bbox.south], z)
    tx0, tx1 = int(xa // TILE), int(xb // TILE)
    ty0, ty1 = int(ya // TILE), int(yb // TILE)
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    jobs = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    log.info("imagery %s z%d: %d tiles (%dx%d)", src.name, z, len(jobs), nx, ny)
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(lambda j: _fetch_tile(src, z, j[0], j[1], cache), jobs))
    img = np.zeros((ny * TILE, nx * TILE, 3), dtype=np.uint8)
    missing = 0
    for (x, y), p in zip(jobs, paths):
        tile = None
        if p is not None:
            try:
                tile = np.asarray(Image.open(io.BytesIO(p.read_bytes())).convert("RGB"))
            except Exception:  # corrupted cache entry
                p.unlink(missing_ok=True)
        if tile is None or tile.shape[:2] != (TILE, TILE) or p.stat().st_size < 3000 and tile.std() < 12:
            missing += 1  # absent or "no data" placeholder tile
            continue
        r, c = (y - ty0) * TILE, (x - tx0) * TILE
        img[r:r + TILE, c:c + TILE] = tile
    grid = RasterGrid(z, tx0 * TILE, ty0 * TILE, nx * TILE, ny * TILE)
    return img, grid, missing / max(1, len(jobs))


def ground_resolution(src: TileSource, lat: float, zoom: int | None = None) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2 ** (zoom or src.zoom))

"""Geodesy helpers: Web Mercator pixel math, local metric CRS, raster grid."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

TILE = 256


def lonlat_to_gpx(lon, lat, z: int):
    """Lon/lat (deg) to global Web Mercator pixel coordinates at zoom z."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    n = TILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def gpx_to_lonlat(x, y, z: int):
    n = TILE * (2 ** z)
    lon = np.asarray(x, dtype=np.float64) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * np.asarray(y, dtype=np.float64) / n))))
    return lon, lat


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


@dataclass
class BBox:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def parse(cls, s: str) -> "BBox":
        w, so, e, n = (float(v) for v in s.split(","))
        return cls(w, so, e, n)

    @property
    def center(self):
        return (self.west + self.east) / 2.0, (self.south + self.north) / 2.0

    def contains(self, lon, lat):
        return (lon >= self.west) & (lon <= self.east) & (lat >= self.south) & (lat <= self.north)

    def expand_m(self, m: float) -> "BBox":
        dlat = m / 111_320.0
        dlon = m / (111_320.0 * math.cos(math.radians(self.center[1])))
        return BBox(self.west - dlon, self.south - dlat, self.east + dlon, self.north + dlat)

    def area_km2(self) -> float:
        h = (self.north - self.south) * 111.32
        w = (self.east - self.west) * 111.32 * math.cos(math.radians(self.center[1]))
        return w * h

    def split(self, max_km: float) -> list["BBox"]:
        """Split into a grid of sub-boxes whose sides are at most max_km."""
        h = (self.north - self.south) * 111.32
        w = (self.east - self.west) * 111.32 * math.cos(math.radians(self.center[1]))
        nx, ny = max(1, math.ceil(w / max_km)), max(1, math.ceil(h / max_km))
        xs = np.linspace(self.west, self.east, nx + 1)
        ys = np.linspace(self.south, self.north, ny + 1)
        return [BBox(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(ny) for i in range(nx)]

    def key(self) -> str:
        return f"{self.west:.5f}_{self.south:.5f}_{self.east:.5f}_{self.north:.5f}"


class Projector:
    """Lon/lat <-> local metric CRS (UTM zone of the area)."""

    def __init__(self, lon: float, lat: float):
        self.crs = utm_crs(lon, lat)
        self._fwd = Transformer.from_crs(4326, self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, 4326, always_xy=True)

    def to_xy(self, lon, lat):
        return self._fwd.transform(lon, lat)

    def to_lonlat(self, x, y):
        return self._inv.transform(x, y)


@dataclass
class RasterGrid:
    """A window of the global Web Mercator pixel grid at zoom z."""

    z: int
    x0: int  # global pixel of the left column
    y0: int  # global pixel of the top row
    w: int
    h: int

    def lonlat_to_px(self, lon, lat):
        x, y = lonlat_to_gpx(lon, lat, self.z)
        return x - self.x0, y - self.y0

    def px_to_lonlat(self, col, row):
        return gpx_to_lonlat(np.asarray(col) + self.x0, np.asarray(row) + self.y0, self.z)

    def m_per_px(self, lat: float) -> float:
        return 156543.03392804097 * math.cos(math.radians(lat)) / (2 ** self.z)

"""Preset study areas. Any other area can be processed with --bbox west,south,east,north."""
from __future__ import annotations

from dataclasses import dataclass

from .geo import BBox


@dataclass(frozen=True)
class City:
    key: str
    label: str
    bbox: BBox
    source: str
    country: str


CITIES = {c.key: c for c in [
    # Prati, Centro Storico, Esquilino, Trastevere, Testaccio, San Giovanni
    City("roma", "Roma (centro, ~30 km2)", BBox(12.440, 41.870, 12.520, 41.920), "esri_wayback", "IT"),
    City("roma_prati", "Roma, Prati (test area, ~1 km2)", BBox(12.455, 41.904, 12.468, 41.912), "esri_wayback", "IT"),
    City("roma_gra", "Roma inside the GRA ring road (~500 km2, long run)",
         BBox(12.36, 41.79, 12.62, 41.99), "esri_wayback", "IT"),
    City("amsterdam", "Amsterdam centre (8 cm open aerial imagery)", BBox(4.870, 52.360, 4.910, 52.380), "pdok", "NL"),
]}

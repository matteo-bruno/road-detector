"""Write results in compact, analysis-ready formats."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

SEGMENT_COLUMNS = [
    # identity and matching to the known street network
    "segment_id", "osm_way_id", "name", "ref", "highway", "from_node", "to_node",
    "from_intersection", "to_intersection", "housenumber_min", "housenumber_max", "n_addresses",
    "length_m", "from_lon", "from_lat", "to_lon", "to_lat", "mid_lon", "mid_lat",
    # measured on imagery
    "measured", "quality", "confidence", "fail_reason",
    "carriageway_width_m", "carriageway_width_iqr_m", "travel_width_m", "lanes", "lanes_source", "lanes_est",
    "parking_sides", "parking_left_width_m", "parking_right_width_m", "parking_left_type",
    "parking_right_type", "parking_left_occupancy", "parking_right_occupancy",
    "sidewalk_sides", "sidewalk_left_width_m", "sidewalk_right_width_m",
    "building_to_building_m", "road_area_m2", "axis_offset_m", "canopy_frac", "shadow_frac",
    # from OpenStreetMap, for comparison and enrichment
    "oneway", "carriageways", "twin_segment_id", "lanes_osm", "maxspeed_kmh", "maxspeed_source",
    "width_osm_m", "sidewalk_osm", "parking_osm_left", "parking_osm_right", "surface_osm",
    "lit_osm", "cycleway_osm", "bus_lane_osm", "bridge", "tunnel", "roundabout", "junction_internal",
    # optional traffic enrichment
    "traffic_current_kmh", "traffic_freeflow_kmh", "traffic_ratio",
]

INTERSECTION_COLUMNS = [
    "intersection_id", "kind", "lon", "lat", "n_arms", "n_streets", "street_names",
    "traffic_signals", "n_crossings", "mean_arm_width_m", "max_arm_width_m", "approx_diameter_m",
    "paved_area_m2", "n_osm_nodes", "osm_node_ids",
]


def _round(df):
    for c in df.columns:
        if c != "geometry" and df[c].dtype.kind == "f":
            df[c] = df[c].round(6 if ("lon" in c or "lat" in c) else 2)
    return df


def write(seg: gpd.GeoDataFrame, inter: gpd.GeoDataFrame, meta: dict, out_dir: Path, geojson: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    ends = shapely.get_coordinates(shapely.get_point(seg.geometry.values, 0))
    seg["from_lon"], seg["from_lat"] = ends[:, 0], ends[:, 1]
    ends = shapely.get_coordinates(shapely.get_point(seg.geometry.values, -1))
    seg["to_lon"], seg["to_lat"] = ends[:, 0], ends[:, 1]
    seg = seg.copy()
    seg["geometry"] = shapely.set_precision(seg.geometry.values, 1e-6)
    s = _round(seg[[c for c in SEGMENT_COLUMNS if c in seg.columns] + ["geometry"]].copy())
    x = _round(inter[[c for c in INTERSECTION_COLUMNS if c in inter.columns] + ["geometry"]].copy())
    s.to_parquet(out_dir / "segments.parquet", compression="zstd", index=False)
    x.to_parquet(out_dir / "intersections.parquet", compression="zstd", index=False)
    s.drop(columns="geometry").to_csv(out_dir / "segments.csv", index=False)
    x.drop(columns="geometry").to_csv(out_dir / "intersections.csv", index=False)
    if geojson:
        s.to_file(out_dir / "segments.geojson", driver="GeoJSON")
        x.to_file(out_dir / "intersections.geojson", driver="GeoJSON")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return s, x


def read(out_dir: Path):
    return (gpd.read_parquet(out_dir / "segments.parquet"),
            gpd.read_parquet(out_dir / "intersections.parquet"),
            json.loads((out_dir / "metadata.json").read_text()))

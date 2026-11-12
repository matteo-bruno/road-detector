import numpy as np
import pytest

from roaddet import measure, tags
from roaddet.geo import BBox, Projector, gpx_to_lonlat, lonlat_to_gpx
from roaddet.network import build_network
from roaddet.stats import describe
import pandas as pd


def test_mercator_roundtrip():
    x, y = lonlat_to_gpx(12.48, 41.9, 19)
    lon, lat = gpx_to_lonlat(x, y, 19)
    assert lon == pytest.approx(12.48, abs=1e-9) and lat == pytest.approx(41.9, abs=1e-9)


def test_tag_parsing():
    assert tags.parse_maxspeed({"maxspeed": "30"}) == (30, "osm")
    assert tags.parse_maxspeed({"maxspeed": "IT:urban"}) == (50, "osm_implicit")
    assert tags.parse_maxspeed({"highway": "residential"}) == (50, "default")
    assert tags.parse_oneway({"oneway": "-1"}) == "backward"
    assert tags.parse_oneway({"junction": "roundabout"}) == "forward"
    assert tags.parse_sidewalk({"sidewalk": "left"}) == "left"
    assert tags.parse_sidewalk({"sidewalk:both": "yes"}) == "both"
    assert tags.parse_sidewalk({}) is None
    assert tags.parse_parking({"parking:both": "lane", "parking:both:orientation": "parallel"}) == ("parallel", "parallel")
    assert tags.parse_parking({"parking:lane:left": "no_parking", "parking:lane:right": "diagonal"}) == ("no", "diagonal")
    assert tags.road_attributes({"highway": "primary", "lanes": "3", "width": "10,5 m"})["width_osm_m"] == 10.5


def _way(wid, nodes, coords, **tg):
    return {"type": "way", "id": wid, "nodes": nodes,
            "geometry": [{"lon": lon, "lat": lat} for lon, lat in coords],
            "tags": {"highway": "residential", **tg}}


def test_network_splits_at_intersections():
    # a "plus" crossing: way 1 runs west-east through node 3, way 2 runs south-north through it
    d = 0.001
    elems = [
        _way(1, [1, 2, 3, 4], [(12.0 - 2 * d, 42.0), (12.0 - d, 42.0), (12.0, 42.0), (12.0 + d, 42.0)], name="Via A"),
        _way(2, [5, 3, 6], [(12.0, 42.0 - d), (12.0, 42.0), (12.0, 42.0 + d)], name="Via B"),
    ]
    bbox = BBox(11.99, 41.99, 12.01, 42.01)
    seg, inter = build_network(elems, Projector(12.0, 42.0), bbox)
    assert len(seg) == 4
    assert set(seg["segment_id"]) == {"1_0", "1_1", "2_0", "2_1"}
    assert len(inter) == 1
    x = inter.iloc[0]
    assert x["n_arms"] == 4 and x["street_names"] == "Via A; Via B"
    a = seg.set_index("segment_id").loc["1_0"]
    assert a["from_node"] == 1 and a["to_node"] == 3 and a["to_intersection"] == x["intersection_id"]
    assert a["length_m"] == pytest.approx(165.6, abs=1.0)


def test_measure_synthetic_street():
    # 10 m asphalt carriageway with a 2 m row of parked cars on the low-t side, lighter surroundings
    p = measure.StripParams()
    t = np.arange(-p.half_width_m, p.half_width_m + 1e-6, p.dt)
    s = np.arange(0, 100.0, p.ds)
    rgb = np.full((len(s), len(t), 3), 200, np.uint8)
    prob = np.zeros((len(s), len(t)), np.uint8)
    road = (t >= -5) & (t <= 5)
    rgb[:, road] = 90
    prob[:, road] = 240
    rng = np.random.default_rng(0)
    cols = np.flatnonzero((t >= -5) & (t < -3))
    for k in range(0, len(s) - 9, 12):  # 4.5 m cars every 6 m
        rgb[k:k + 9][:, cols] = rng.choice([20, 235, 160])
    bld = np.zeros_like(prob)
    bld[:, (t < -8) | (t > 9)] = 1
    res = measure.measure_strip(rgb, prob, bld, s, t, p, 5.0, 5.0)
    assert res["measured"]
    assert res["carriageway_width_m"] == pytest.approx(10.0, abs=0.6)
    assert res["parking_right_width_m"] == pytest.approx(2.0, abs=0.6)
    assert res["parking_left_width_m"] == 0.0
    assert res["sidewalk_left_width_m"] == pytest.approx(4.0, abs=0.6)
    assert res["sidewalk_right_width_m"] == pytest.approx(3.0, abs=0.6)


def test_describe_weighted():
    d = describe(pd.Series([10.0, 20.0]), pd.Series([300.0, 100.0]))
    assert d["mean"] == 15.0 and d["mean_length_weighted"] == 12.5 and d["n"] == 2

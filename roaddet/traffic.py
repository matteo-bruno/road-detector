"""Optional traffic enrichment with the TomTom Traffic Flow API.

Needs an API key in the TOMTOM_API_KEY environment variable. For each selected segment the
flow at its midpoint is queried: current speed, free-flow speed and their ratio.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import requests

log = logging.getLogger(__name__)
URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/14/json"
MAIN = ("motorway", "trunk", "primary", "secondary", "tertiary")


def enrich(seg, classes=MAIN, max_requests: int = 2000, pause: float = 0.05):
    key = os.environ.get("TOMTOM_API_KEY")
    if not key:
        raise RuntimeError("set TOMTOM_API_KEY to use the traffic enrichment")
    for c in ("traffic_current_kmh", "traffic_freeflow_kmh", "traffic_ratio"):
        seg[c] = np.nan
    sel = seg[seg["highway"].str.replace("_link", "").isin(classes) & ~seg["junction_internal"]]
    sel = sel.sort_values("length_m", ascending=False).head(max_requests)
    s = requests.Session()
    for i, r in sel.iterrows():
        try:
            resp = s.get(URL, params={"point": f"{r['mid_lat']},{r['mid_lon']}", "unit": "KMPH", "key": key}, timeout=20)
            d = resp.json().get("flowSegmentData") if resp.status_code == 200 else None
        except (requests.RequestException, ValueError):
            d = None
        if d and d.get("freeFlowSpeed"):
            seg.at[i, "traffic_current_kmh"] = d["currentSpeed"]
            seg.at[i, "traffic_freeflow_kmh"] = d["freeFlowSpeed"]
            seg.at[i, "traffic_ratio"] = round(d["currentSpeed"] / d["freeFlowSpeed"], 2)
        time.sleep(pause)
    log.info("traffic: %d of %d segments enriched", int(seg["traffic_ratio"].notna().sum()), len(sel))
    return seg

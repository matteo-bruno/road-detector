# road-detector

Detects and measures city streets on satellite or aerial imagery and matches them to the
known street network (OpenStreetMap). For every street segment it writes the street name,
civic number range, start and end nodes, length, carriageway width and, where visible,
parking rows, sidewalks, lanes, plus the attributes OSM already knows (one-way, lanes, speed
limit, surface). Intersections are separate objects. City statistics (mean, median,
percentiles of the widths, by road class) are produced from the result.

First use case: Rome. Results for the central area are in [`data/roma`](data/roma), the
statistics in [`data/roma/REPORT.md`](data/roma/REPORT.md).

## Install and run

```bash
pip install -e .
```

```bash
roaddet cities
```

```bash
roaddet run --city roma --debug 40
```

Any other area, with any imagery source:

```bash
roaddet run --bbox 9.18,45.46,9.20,45.47 --source esri --country IT --out data/milano_test
```

Other commands: `roaddet stats data/roma` rebuilds statistics and report from saved results,
`roaddet traffic data/roma` adds TomTom traffic (needs `TOMTOM_API_KEY`). Useful `run`
options: `--include service pedestrian`, `--geojson`, `--reuse-surface` (re-measure without
recomputing the surface raster), `--chunk-km`.

Downloads (imagery tiles, OSM extracts) are cached in `cache/`, which is not committed.

## How it works

1. **Street network** (`osm.py`, `network.py`). Drivable OSM ways are downloaded through
   Overpass and split at junction nodes. A *segment* is the piece of a way between two
   consecutive junctions, so it has a start node and an end node. Junction nodes with three
   or more arms become *intersections*; the nodes of one roundabout, or nodes joined by links
   shorter than 15 m (dual carriageway crossings), are merged into one intersection object.
   Antiparallel one-way segments with the same name are flagged as dual carriageways.
2. **Imagery** (`imagery.py`). XYZ tiles are mosaicked per 2 km chunk. Sources are pluggable:
   `esri_wayback` (Esri World Imagery, release of 2021-11-30, 0.22 m/pixel in Rome, chosen
   because it has diffuse light and almost no cast shadows), `esri` (current release) and
   `pdok` (Netherlands, open 8 cm aerial imagery).
3. **Road surface detection** (`surface.py`). A gradient boosting pixel classifier (colour,
   local texture, gradients) is trained for each run with weak labels taken from OSM: pixels on
   road centre lines are positives, pixels inside building footprints or far from any road are
   negatives. No manual annotation is needed, so the model adapts to each city and image set.
4. **Cross-section measurement** (`measure.py`). Each segment is straightened: the image is
   resampled on a grid that follows the centre line (0.5 m along, 0.25 m across, 22 m each
   side). On this strip, in 20 m blocks:
   - the carriageway is the continuous band of road surface around the axis; the OSM axis may
     be off by up to 5 m, the band is searched, not assumed symmetric;
   - a parking row is a band at the margin where many pixels differ from the asphalt
     brightness and brightness alternates along the axis (cars and gaps); its width gives the
     type (parallel, diagonal, perpendicular) and the share of non-asphalt pixels the occupancy;
   - the sidewalk is the space between the kerb and the first building footprint (OSM);
   - the segment value is the median over blocks, the interquartile range feeds `confidence`.
   The first and last 10 m next to an intersection are ignored.
5. **Enrichment** (`tags.py`, `pipeline.py`, `traffic.py`). OSM tags are normalised and kept
   next to the measured values: one-way, lanes, speed limit (with source), sidewalk, parking,
   surface, width. Civic numbers come from OSM addresses with the same street name. Traffic
   is optional (TomTom Flow API).
6. **Statistics** (`stats.py`). `stats.json`, `REPORT.md` and three charts.

## Output

| file | content |
|---|---|
| `segments.parquet` | GeoParquet (zstd), one row per segment with geometry, EPSG:4326 |
| `segments.csv` | same table without geometry (end and mid point coordinates are columns) |
| `intersections.parquet`, `intersections.csv` | one row per intersection |
| `stats.json`, `REPORT.md`, `*.png` | statistics, report, charts |
| `metadata.json` | area, imagery, resolution, run date |
| `debug/*.jpg` | straightened strips with detected edges, for visual checks |

```python
import geopandas as gpd
seg = gpd.read_parquet("data/roma/segments.parquet")
seg[seg.quality != "not_measured"].groupby("highway").carriageway_width_m.median()
```

### Segment columns

| column | meaning |
|---|---|
| `segment_id` | `<osm way id>_<index along the way>` |
| `osm_way_id`, `name`, `ref`, `highway` | matched OSM street |
| `from_node`, `to_node` | OSM node ids at the two ends |
| `from_intersection`, `to_intersection` | intersection ids, empty if the end is not an intersection |
| `housenumber_min`, `housenumber_max`, `n_addresses` | civic number range matched to the segment |
| `length_m` | length along the centre line |
| `measured`, `quality`, `confidence`, `fail_reason` | whether the imagery gave a measurement, `good` / `fair` / `poor`, score 0 to 1, reason when not measured (`canopy`, `open_area`, `too_short`, `tunnel`, `junction`, `not_visible`) |
| `carriageway_width_m` | kerb to kerb, parking rows included |
| `carriageway_width_iqr_m` | spread of the width between blocks |
| `travel_width_m` | carriageway minus parking rows |
| `lanes`, `lanes_source`, `lanes_est` | OSM lanes when tagged, otherwise estimated as travel width / 3.5 m |
| `parking_sides`, `parking_<side>_width_m`, `parking_<side>_type`, `parking_<side>_occupancy` | parking rows; side is left or right of the digitised direction |
| `sidewalk_sides`, `sidewalk_<side>_width_m` | kerb to building distance, 0.8 to 8 m |
| `building_to_building_m` | facade to facade distance |
| `road_area_m2` | carriageway width x length |
| `axis_offset_m`, `canopy_frac`, `shadow_frac` | diagnostics |
| `oneway` | `forward`, `backward` or `both` |
| `carriageways`, `twin_segment_id` | 2 when the street has a separate opposite carriageway |
| `lanes_osm`, `maxspeed_kmh`, `maxspeed_source`, `width_osm_m`, `sidewalk_osm`, `parking_osm_left`, `parking_osm_right`, `surface_osm`, `lit_osm`, `cycleway_osm`, `bus_lane_osm`, `bridge`, `tunnel`, `roundabout` | OSM attributes; `maxspeed_source` is `osm`, `osm_implicit` or `default` (urban default) |
| `traffic_current_kmh`, `traffic_freeflow_kmh`, `traffic_ratio` | only with the TomTom enrichment |

### Intersection columns

`intersection_id`, `kind` (junction or roundabout), `lon`, `lat`, `n_arms`, `n_streets`,
`street_names`, `traffic_signals`, `n_crossings`, `mean_arm_width_m`, `max_arm_width_m`,
`approx_diameter_m`, `paved_area_m2`, `n_osm_nodes`, `osm_node_ids`.

## Reliability and limits

The imagery over Rome is satellite imagery at about 0.3 m real resolution, which sets what can
be trusted:

- **Length, nodes, names, one-way, speed limits**: from OSM, reliable.
- **Carriageway width**: measured, typical block to block spread 1 to 2 m. It is biased
  upwards where the sidewalk is paved in asphalt of the same colour, and it is not measured
  under continuous tree canopy.
- **Parking rows**: heuristic. Presence is fairly reliable on streets with visible cars, the
  width is approximate (a row is 2 to 5 m and a pixel is 0.22 m), occupancy refers to the
  image date.
- **Sidewalks**: indirect (kerb to OSM building footprint), so they inherit the positional
  error of footprints and the building lean of the image. Use the sum of both sides, or
  `building_to_building_m - carriageway_width_m`, rather than a single side.
- **Lanes**: OSM where tagged. The estimate from the width is a capacity, not painted lanes.
- **Directions, speed limits, traffic**: not visible from above, they come from OSM / TomTom.
- No surveyed ground truth was available, so the checks are internal consistency, agreement
  with OSM tags (see the report) and visual inspection of the debug strips.

With sharper imagery (for example `--source pdok` in the Netherlands) the same code measures
kerbs and parking rows more precisely.

## Data sources and licences

Street network, buildings and addresses: OpenStreetMap contributors, ODbL. Imagery: Esri World
Imagery (Esri, Maxar, Earthstar Geographics), used under Esri terms of use; tiles are not
redistributed here, only derived measurements. PDOK aerial imagery: CC BY 4.0.

## Tests

```bash
pytest -q
```

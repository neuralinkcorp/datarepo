from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

try:
    import geopandas as gpd
    import pyarrow.parquet as pq
    import shapely.geometry as sg
    from pyproj import CRS, Transformer
    import polars as pl
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

CAPTURES = Path("kwf-field-data/captures")
DERIVED = Path("kwf-field-data/derived")


def _require_geo():
    if not HAS_GEO:
        raise ImportError("Install requirements-geospatial.txt")


def _stable_hash(obj: dict) -> str:
    """SHA-256 of canonical JSON."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _vertex_to_point(v: dict) -> "sg.Point | None":
    """Return a WGS-84 Point only if crs is EPSG:4326."""
    if v.get("crs") != "EPSG:4326":
        return None
    x, y = v.get("x"), v.get("y")
    if x is None or y is None:
        return None
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (-180 <= x <= 180 and -90 <= y <= 90):
        return None
    z = v.get("z")
    if z is not None:
        try:
            z = float(z)
            return sg.Point(x, y, z)
        except (TypeError, ValueError):
            pass
    return sg.Point(x, y)


def _make_geometry(obs: dict) -> "sg.BaseGeometry | None":
    """Build WGS-84 geometry from vertices, or None if not all EPSG:4326."""
    _require_geo()
    vertices = obs.get("vertices", [])
    shape = obs.get("shape", "Point")

    points = [_vertex_to_point(v) for v in vertices]
    if any(p is None for p in points):
        return None  # source-only; pending normalization

    if shape == "Point":
        return points[0] if points else None
    elif shape == "LineString":
        coords = [(p.x, p.y) + ((p.z,) if p.has_z else ()) for p in points]
        return sg.LineString(coords) if len(coords) >= 2 else None
    elif shape == "Polygon":
        coords = [(p.x, p.y) + ((p.z,) if p.has_z else ()) for p in points]
        if len(set((c[0], c[1]) for c in coords)) >= 3:
            return sg.Polygon(coords)
        return None
    return None


def normalize_observation(obs: dict, project: dict, source_path: Path) -> dict:
    """
    Normalize one observation dict to a flat row for GeoParquet.
    Returns a dict with keys matching the parquet schema.
    Does NOT invent coordinates, accuracy, or survey status.
    """
    _require_geo()
    obs_id = obs.get("id", "")
    revision = obs.get("revision", 1)
    geom = _make_geometry(obs)

    vertices = obs.get("vertices", [])
    all_wgs84 = all(v.get("crs") == "EPSG:4326" for v in vertices)

    # Bounding box columns (only for WGS-84 observations)
    min_lon = max_lon = min_lat = max_lat = None
    if geom is not None and all_wgs84:
        b = geom.bounds  # (minx, miny, maxx, maxy)
        min_lon, min_lat, max_lon, max_lat = b[0], b[1], b[2], b[3]

    source_wkb = None
    if geom is not None:
        import shapely.wkb as swkb
        source_wkb = swkb.dumps(geom, include_srid=False)

    return {
        "id": obs_id,
        "revision": revision,
        "project_id": project.get("id", ""),
        "type": obs.get("type", ""),
        "subtype": obs.get("subtype", ""),
        "shape": obs.get("shape", "Point"),
        "source_label": obs.get("sourceLabel", obs.get("source_label", "")),
        "notes": obs.get("notes", ""),
        "soil_ph": obs.get("soilPh", obs.get("soil_ph")),
        "moisture": obs.get("moisture"),
        "moisture_unit": obs.get("moistureUnit", obs.get("moisture_unit")),
        "surveyed": obs.get("surveyed", False),
        "survey_reference": obs.get("surveyReference", obs.get("survey_reference")),
        "gcp_label": obs.get("gcpLabel", obs.get("gcp_label")),
        "pending_normalization": not all_wgs84,
        "source_path": str(source_path),
        "geometry": geom,
        "source_wkb": source_wkb,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
    }


def load_latest(root: Path) -> tuple[list[tuple[dict, Path]], dict[str, str]]:
    """
    Walk root/**/source-observations.json files.
    For each observation ID, keep the highest revision.
    Raise ValueError on conflicting content at the same revision.
    Returns (rows, hashes) where hashes maps id -> content_hash.
    """
    best: dict[str, tuple[int, dict, Path]] = {}
    hashes: dict[str, str] = {}

    for path in sorted(root.rglob("source-observations.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        project = data.get("project", {})
        for obs in data.get("observations", []):
            obs_id = obs.get("id")
            if not obs_id:
                continue
            rev = obs.get("revision", 1)
            h = _stable_hash(obs)

            if obs_id in best:
                existing_rev, existing_obs, existing_path = best[obs_id]
                if rev == existing_rev:
                    if hashes.get(obs_id) != h:
                        raise ValueError(
                            f"Conflicting revision {rev} for observation {obs_id}: "
                            f"{existing_path} vs {path}"
                        )
                elif rev > existing_rev:
                    best[obs_id] = (rev, obs, path)
                    hashes[obs_id] = h
            else:
                best[obs_id] = (rev, obs, path)
                hashes[obs_id] = h

    rows = [(obs, path) for (_, obs, path) in best.values()]
    return rows, hashes


def normalize_archive(root: Path, output: Path) -> "gpd.GeoDataFrame":
    """
    Load all source-observations.json under root, deduplicate by revision,
    and write observations.parquet with GeoParquet metadata.
    """
    _require_geo()
    import geopandas as gpd
    import shapely.wkb as swkb

    rows_and_paths, _ = load_latest(root)
    if not rows_and_paths:
        # Return empty GeoDataFrame with correct schema
        gdf = gpd.GeoDataFrame(
            columns=["id","revision","project_id","type","subtype","shape",
                     "source_label","notes","soil_ph","moisture","moisture_unit",
                     "surveyed","survey_reference","gcp_label","pending_normalization",
                     "source_path","geometry","source_wkb","min_lon","max_lon",
                     "min_lat","max_lat"],
            geometry="geometry",
            crs="EPSG:4326"
        )
        output.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(output / "observations.parquet")
        return gdf

    # We need a project dict per observation — read from the file
    all_rows = []
    for path in sorted(root.rglob("source-observations.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        project = data.get("project", {})
        for obs in data.get("observations", []):
            all_rows.append((obs, project, path))

    # Deduplicate by observation id keeping highest revision
    best: dict[str, tuple[int, dict, dict, Path]] = {}
    for obs, project, path in all_rows:
        obs_id = obs.get("id")
        if not obs_id:
            continue
        rev = obs.get("revision", 1)
        h = _stable_hash(obs)
        if obs_id in best:
            existing_rev = best[obs_id][0]
            if rev > existing_rev:
                best[obs_id] = (rev, obs, project, path)
        else:
            best[obs_id] = (rev, obs, project, path)

    normalized = []
    for rev, obs, project, path in best.values():
        row = normalize_observation(obs, project, path)
        normalized.append(row)

    gdf = gpd.GeoDataFrame(normalized, geometry="geometry", crs="EPSG:4326")

    # Ensure source_wkb is bytes
    gdf["source_wkb"] = gdf["source_wkb"].apply(
        lambda v: bytes(v) if v is not None else None
    )

    output.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output / "observations.parquet")
    return gdf


def catalog_observations() -> "gpd.GeoDataFrame":
    """Entry point for kwf_catalog.py table registration."""
    _require_geo()
    path = DERIVED / "observations.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python kwf_geo.py normalize"
        )
    import geopandas as gpd
    return gpd.read_parquet(path)


def export_gcps(
    root: Path,
    output: Path,
    project_id: str,
    target_crs_id: str,
    target_vcrs_id: str,
    site_boundary_path: Path,
    metric_crs_id: str,
    minimum_points: int = 5,
    minimum_hull_area_m2: float = 10.0,
    minimum_separation_m: float = 1.0,
) -> Path:
    """
    Export surveyed GCP observations for a project.
    Requires explicit CRSs. Rejects coordinate-epoch input.
    Does not guess CRS or height reference.
    """
    _require_geo()
    import geopandas as gpd
    import shapely.geometry as sg
    import shapely.wkb as swkb

    # Validate CRSs upfront
    try:
        target_crs = CRS.from_user_input(target_crs_id)
    except Exception as e:
        raise ValueError(f"Invalid target CRS '{target_crs_id}': {e}")

    try:
        target_vcrs = CRS.from_user_input(target_vcrs_id)
    except Exception as e:
        raise ValueError(f"Invalid target vertical CRS '{target_vcrs_id}': {e}")

    try:
        metric_crs = CRS.from_user_input(metric_crs_id)
    except Exception as e:
        raise ValueError(f"Invalid metric CRS '{metric_crs_id}': {e}")

    # Reject coordinate-epoch input (time-dependent transforms not yet verified)
    for crs_obj, name in [(target_crs, "target"), (target_vcrs, "vertical"), (metric_crs, "metric")]:
        wkt = crs_obj.to_wkt()
        if "coordinateEpoch" in wkt or "EPOCH" in wkt.upper():
            raise ValueError(
                f"Coordinate-epoch CRS not supported for {name} CRS. "
                "Time-dependent transformations are not yet verified."
            )

    # Load site boundary
    if not site_boundary_path.exists():
        raise FileNotFoundError(f"Site boundary not found: {site_boundary_path}")
    site_gdf = gpd.read_file(site_boundary_path)
    site_boundary = site_gdf.geometry.union_all()

    # Load all observations for this project
    rows_and_paths, _ = load_latest(root)

    gcps = []
    for obs, path in rows_and_paths:
        if obs.get("project_id", obs.get("projectId", "")) != project_id:
            # Also check inside the file's project block
            pass
        if not obs.get("surveyed", False):
            continue
        if not obs.get("type", "").upper() == "GCP":
            continue
        vertices = obs.get("vertices", [])
        if len(vertices) != 1:
            continue
        v = vertices[0]
        if v.get("crs") != "EPSG:4326":
            raise ValueError(
                f"GCP {obs.get('id')} has non-WGS84 source coordinate. "
                "Cannot export without normalization."
            )
        if v.get("vcrs", "unknown") == "unknown":
            raise ValueError(
                f"GCP {obs.get('id')} has unknown height reference. "
                "Cannot export without a known vertical CRS."
            )
        pt = _vertex_to_point(v)
        if pt is None:
            continue
        gcps.append((obs, v, pt))

    if len(gcps) < minimum_points:
        raise ValueError(
            f"Insufficient GCPs: {len(gcps)} found, {minimum_points} required."
        )

    # Check spatial distribution using metric CRS
    gdf_metric = gpd.GeoDataFrame(
        [{"geometry": pt} for _, _, pt in gcps],
        geometry="geometry",
        crs="EPSG:4326"
    ).to_crs(metric_crs_id)

    from shapely.ops import unary_union
    hull = gdf_metric.geometry.unary_union.convex_hull
    if hull.area < minimum_hull_area_m2:
        raise ValueError(
            f"GCP hull area {hull.area:.1f} m² < minimum {minimum_hull_area_m2} m²."
        )

    # Check minimum separation
    warnings = []
    coords_metric = [(g.x, g.y) for g in gdf_metric.geometry]
    for i in range(len(coords_metric)):
        for j in range(i+1, len(coords_metric)):
            dx = coords_metric[i][0] - coords_metric[j][0]
            dy = coords_metric[i][1] - coords_metric[j][1]
            dist = (dx**2 + dy**2)**0.5
            if dist < minimum_separation_m:
                warnings.append(
                    f"GCPs {gcps[i][0].get('id')} and {gcps[j][0].get('id')} "
                    f"are {dist:.2f} m apart (< {minimum_separation_m} m minimum)."
                )

    # Build target CRS (compound if needed)
    target_horizontal = CRS.from_user_input(target_crs_id)

    metadata = []
    for obs, v, pt in gcps:
        metadata.append({
            "id": obs.get("id"),
            "gcpLabel": obs.get("gcpLabel", obs.get("gcp_label")),
            "revision": obs.get("revision", 1),
            "sourceCrs": v.get("crs"),
            "sourceX": v.get("x"),
            "sourceY": v.get("y"),
            "sourceZ": v.get("z"),
            "sourceVcrs": v.get("vcrs"),
            "horizontalAccuracyM": v.get("horizontalAccuracyM"),
            "verticalAccuracyM": v.get("verticalAccuracyM"),
            "surveyReference": obs.get("surveyReference", obs.get("survey_reference")),
        })

    directory = output / project_id
    directory.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schemaVersion": 1,
        "projectId": project_id,
        "targetCrs": target_crs_id,
        "targetVcrs": target_vcrs_id,
        "targetAxisUnits": [
            axis.unit_name for axis in target_crs.axis_info
        ],
        "horizontalAccuracyUnit": (
            "meter" if target_horizontal.is_geographic
            else target_horizontal.axis_info[0].unit_name
        ),
        "verticalAccuracyUnit": target_vcrs.axis_info[0].unit_name if target_vcrs.axis_info else "meter",
        "qualityCheckCrs": metric_crs_id,
        "policy": {
            "minimumPoints": minimum_points,
            "minimumHullAreaM2": minimum_hull_area_m2,
            "minimumSeparationM": minimum_separation_m,
        },
        "points": metadata,
        "warnings": warnings,
        "imagePixelCorrespondencesIncluded": False,
    }

    (directory / "gcp_export.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    return directory


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    normalize = commands.add_parser("normalize")
    normalize.add_argument("--captures", type=Path, default=CAPTURES)
    normalize.add_argument("--output", type=Path, default=DERIVED)

    gcps = commands.add_parser("gcps")
    gcps.add_argument("--captures", type=Path, default=CAPTURES)
    gcps.add_argument("--output", type=Path, default=DERIVED / "gcp-exports")
    gcps.add_argument("--project-id", required=True)
    gcps.add_argument("--target-crs", required=True)
    gcps.add_argument("--target-vcrs", required=True)
    gcps.add_argument("--site-boundary", type=Path, required=True)
    gcps.add_argument("--metric-crs", required=True)
    gcps.add_argument("--minimum-points", type=int, default=5)
    gcps.add_argument("--minimum-hull-area-m2", type=float, default=10.0)
    gcps.add_argument("--minimum-separation-m", type=float, default=1.0)

    arguments = parser.parse_args()

    if arguments.command == "normalize":
        frame = normalize_archive(arguments.captures, arguments.output)
        print(f"Normalized {len(frame)} unique observations")
    else:
        result = export_gcps(
            root=arguments.captures,
            output=arguments.output,
            project_id=arguments.project_id,
            target_crs_id=arguments.target_crs,
            target_vcrs_id=arguments.target_vcrs,
            site_boundary_path=arguments.site_boundary,
            metric_crs_id=arguments.metric_crs,
            minimum_points=arguments.minimum_points,
            minimum_hull_area_m2=arguments.minimum_hull_area_m2,
            minimum_separation_m=arguments.minimum_separation_m,
        )
        print(result)


if __name__ == "__main__":
    main()

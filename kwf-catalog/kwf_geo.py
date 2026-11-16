from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import polars as pl
from pyproj import CRS
from pyproj.transformer import TransformerGroup
from shapely.geometry import LineString, MultiPoint, Point, Polygon, mapping, shape
from shapely.ops import transform as transform_geometry


ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "kwf-field-data" / "captures"
DERIVED = ROOT / "kwf-field-data" / "derived"
DATA_ROOT = ROOT / "kwf-field-data"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def epoch_iso(value):
    if value is None:
        return None
    milliseconds = finite(value, "Observation timestamp")
    try:
        return datetime.fromtimestamp(
            milliseconds / 1000, timezone.utc
        ).isoformat()
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("Observation timestamp is out of range") from error


def resolve_crs(identifier, project=None):
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Missing CRS")
    definition = (project or {}).get("crsRegistry", {}).get(
        identifier, identifier
    )
    if isinstance(definition, dict):
        definition = definition.get("definition")
    try:
        return CRS.from_user_input(definition)
    except Exception as error:
        raise ValueError(f"Unknown CRS: {identifier}") from error


def transformer(source, target):
    group = TransformerGroup(
        source, target, always_xy=True, allow_ballpark=False
    )
    if not group.best_available:
        missing = sorted({
            grid.short_name
            for operation in group.unavailable_operations
            for grid in operation.grids
            if not grid.available
        })
        raise ValueError(
            "Best transformation unavailable. Missing resources: "
            + ", ".join(missing)
        )
    if not group.transformers:
        raise ValueError(
            f"No permitted transformation: {source.name} -> {target.name}"
        )
    return group.transformers[0]


def coordinates_of(geometry):
    if geometry.geom_type in {"Point", "LineString"}:
        return list(geometry.coords)
    if geometry.geom_type == "Polygon":
        return (
            list(geometry.exterior.coords)
            + [coordinate for ring in geometry.interiors for coordinate in ring.coords]
        )
    raise ValueError("Supported geometry: Point, LineString, Polygon")


def source_geometry(observation, project):
    vertices = observation.get("vertices")
    if not isinstance(vertices, list) or not 1 <= len(vertices) <= 1000:
        raise ValueError("Geometry requires 1–1,000 vertices")
    identifiers = {vertex.get("crs") for vertex in vertices}
    if len(identifiers) != 1:
        raise ValueError("One geometry cannot mix source CRSs")
    identifier = next(iter(identifiers))
    crs = resolve_crs(identifier, project)
    if not (crs.is_projected or crs.is_geographic):
        raise ValueError("Expected a horizontal CRS")
    coordinates = [
        (finite(vertex.get("x"), "x"), finite(vertex.get("y"), "y"))
        for vertex in vertices
    ]
    kind = observation.get("shape")
    distinct = len(set(coordinates))
    if kind == "Point":
        if len(coordinates) != 1:
            raise ValueError("Point requires one vertex")
        geometry = Point(coordinates[0])
    elif kind == "LineString":
        if distinct < 2:
            raise ValueError("Line requires two distinct vertices")
        geometry = LineString(coordinates)
    elif kind == "Polygon":
        if distinct < 3:
            raise ValueError("Polygon requires three distinct vertices")
        geometry = Polygon(coordinates)
    else:
        raise ValueError(f"Unsupported shape: {kind}")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid or empty source geometry")
    for vertex in vertices:
        if vertex.get("z") is not None:
            finite(vertex["z"], "z")
        for key in ("horizontalAccuracyM", "verticalAccuracyM"):
            if vertex.get(key) is not None and finite(vertex[key], key) < 0:
                raise ValueError(f"{key} cannot be negative")
    return geometry, identifier, crs


def check_wgs84(geometry):
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid display geometry")
    for coordinate in coordinates_of(geometry):
        x, y = coordinate[:2]
        if not (
            math.isfinite(x) and math.isfinite(y)
            and -180 <= x <= 180 and -90 <= y <= 90
        ):
            raise ValueError("Invalid WGS 84 display coordinates")


def in_area(x, y, crs):
    area = crs.area_of_use
    if area is None:
        return True
    longitude_ok = (
        area.west <= x <= area.east
        if area.west <= area.east
        else x >= area.west or x <= area.east
    )
    return longitude_ok and area.south <= y <= area.north


def study_extent(project):
    extent = project.get("studyExtent")
    if extent is None:
        return None
    if not isinstance(extent, list) or len(extent) != 4:
        raise ValueError("Study extent needs west,south,east,north")
    west, south, east, north = [
        finite(value, "Study extent") for value in extent
    ]
    if not (
        -180 <= west < east <= 180 and -90 <= south < north <= 90
    ):
        raise ValueError("Invalid study extent")
    return west, south, east, north


def safe_asset(snapshot, reference):
    if not isinstance(reference, str):
        raise ValueError("Asset references must be strings")
    candidate = (snapshot.parent / reference).resolve()
    try:
        candidate.relative_to(snapshot.parent.resolve())
    except ValueError as error:
        raise ValueError("Asset reference leaves its snapshot directory") from error
    try:
        relative = candidate.relative_to(DATA_ROOT.resolve())
    except ValueError:
        capture_root = next(
            (parent for parent in snapshot.parents if parent.name == "captures"),
            None,
        )
        if capture_root is None:
            raise ValueError("Asset reference is outside a captures directory")
        relative = candidate.relative_to(capture_root.parent)
    return relative.as_posix()


def legacy_observation(feature, project):
    properties = feature.get("properties") or {}
    geometry = shape(feature.get("geometry"))
    if geometry.geom_type not in {"Point", "LineString", "Polygon"}:
        raise ValueError("Unsupported legacy geometry")
    check_wgs84(geometry)
    if geometry.geom_type == "Polygon" and geometry.interiors:
        raise ValueError("Legacy polygon holes need explicit migration; not discarded")
    points = coordinates_of(geometry)
    if geometry.geom_type == "Polygon":
        points = points[:-1]
    metadata = properties.get("vertexMetadata") or []
    vertices = []
    for index, coordinate in enumerate(points):
        item = metadata[index] if index < len(metadata) else {}
        vertices.append({
            "x": coordinate[0],
            "y": coordinate[1],
            "z": item.get("z", coordinate[2] if len(coordinate) > 2 else None),
            "crs": "EPSG:4326",
            "vcrs": item.get("vcrs", "unknown"),
            "source": item.get("source", "unknown"),
            "horizontalAccuracyM": item.get(
                "horizontalAccuracyM",
                item.get("accuracyMeters", item.get("accuracy")),
            ),
            "verticalAccuracyM": item.get("verticalAccuracyM"),
            "capturedAtEpochMs": item.get("capturedAtEpochMs"),
            "provider": item.get("provider"),
        })
    timestamp = properties.get(
        "observedAtEpochMs", properties.get("capturedAtEpochMs")
    )
    if timestamp is None and properties.get("observedAt"):
        parsed = datetime.fromisoformat(
            properties["observedAt"].replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            raise ValueError("Legacy observation time has no timezone")
        timestamp = int(parsed.timestamp() * 1000)
    record_id = feature.get("id")
    if not record_id:
        raise ValueError("Legacy record has no permanent ID")
    return {
        "id": str(record_id),
        "revision": properties.get("revision", 1),
        "projectId": properties.get("projectId", project["id"]),
        "type": properties.get("type", "Unspecified"),
        "subtype": properties.get("subtype", ""),
        "shape": geometry.geom_type,
        "vertices": vertices,
        "observedAtEpochMs": timestamp,
        "updatedAtEpochMs": properties.get("updatedAtEpochMs"),
        "notes": properties.get("notes", ""),
        "soilPh": properties.get("soilPh"),
        "moisture": properties.get("moisture"),
        "moistureUnit": properties.get("moistureUnit"),
        "surveyed": properties.get("surveyed") is True,
        "surveyReference": properties.get("surveyReference"),
        "gcpLabel": properties.get("gcpLabel"),
        "assets": [
            item.get("path") if isinstance(item, dict) else item
            for item in properties.get("assets", [])
        ],
        "legacyProperties": properties,
    }


def load_latest(root=CAPTURES):
    latest = {}
    projects = {}
    problems = []
    directories = sorted({
        path.parent
        for name in ("source-observations.json", "observations.geojson")
        for path in root.rglob(name)
    })
    for directory in directories:
        source = directory / "source-observations.json"
        display = directory / "observations.geojson"
        try:
            if source.exists():
                document = read_json(source)
                project = document.get("project") or {}
                if not project.get("id"):
                    raise ValueError("Source snapshot has no project ID")
                observations = document.get("observations", [])
                if not isinstance(observations, list):
                    raise ValueError("Snapshot observations must be a list")
                if display.exists():
                    display_document = read_json(display)
                    if display_document.get("type") == "FeatureCollection":
                        display_by_id = {
                            str(feature.get("id")): feature
                            for feature in display_document.get("features", [])
                        }
                        for observation in observations:
                            feature = display_by_id.get(str(observation.get("id")))
                            if feature is None:
                                continue
                            properties = feature.get("properties") or {}
                            fallbacks = {
                                "observedAtEpochMs": properties.get(
                                    "observedAtEpochMs",
                                    properties.get("capturedAtEpochMs"),
                                ),
                                "updatedAtEpochMs": properties.get(
                                    "updatedAtEpochMs"
                                ),
                                "assets": properties.get("assets"),
                                "shape": (feature.get("geometry") or {}).get("type"),
                                "type": properties.get("type"),
                                "subtype": properties.get("subtype"),
                                "surveyed": properties.get("surveyed"),
                                "surveyReference": properties.get("surveyReference"),
                                "gcpLabel": properties.get("gcpLabel"),
                                "notes": properties.get("notes"),
                                "soilPh": properties.get("soilPh"),
                                "moisture": properties.get("moisture"),
                                "moistureUnit": properties.get("moistureUnit"),
                                "sourceLabel": properties.get("sourceLabel"),
                            }
                            for key, value in fallbacks.items():
                                if key not in observation and value is not None:
                                    observation[key] = value
                path = source
            else:
                document = read_json(display)
                if document.get("type") != "FeatureCollection":
                    raise ValueError("Expected FeatureCollection")
                if document.get("crs"):
                    raise ValueError("CRS-tagged legacy GeoJSON requires explicit migration")
                features = document.get("features", [])
                first = (
                    (features[0].get("properties") or {})
                    if features else {}
                )
                project = {
                    "id": first.get("projectId")
                    or "legacy-folder:" + directory.parent.name,
                    "name": first.get("project") or directory.parent.name,
                }
                observations = [
                    legacy_observation(feature, project) for feature in features
                ]
                path = display
            projects[project["id"]] = project
            for observation in observations:
                record_id = observation.get("id")
                if not record_id:
                    raise ValueError("Observation has no ID")
                if observation.get("projectId", project["id"]) != project["id"]:
                    raise ValueError("Observation belongs to another project")
                revision = observation.get("revision", 1)
                if (
                    isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision < 1
                ):
                    raise ValueError("Invalid revision")
                key = (project["id"], str(record_id))
                content_hash = digest(observation)
                previous = latest.get(key)
                if previous is None or revision > previous["revision"]:
                    latest[key] = {
                        "revision": revision,
                        "hash": content_hash,
                        "observation": observation,
                        "project": project,
                        "path": path,
                    }
                elif (
                    revision == previous["revision"]
                    and content_hash != previous["hash"]
                ):
                    raise ValueError(
                        f"Conflicting equal revision: {key}, revision {revision}"
                    )
        except Exception as error:
            problems.append({"snapshot": str(directory), "problem": str(error)})
    return list(latest.values()), projects, problems


def normalize_observation(observation, project, source_path):
    original, source_id, source_crs = source_geometry(observation, project)
    operation = transformer(source_crs, CRS.from_epsg(4326))
    display = transform_geometry(
        lambda x, y, z=None: operation.transform(x, y, errcheck=True),
        original,
    )
    check_wgs84(display)
    warnings = []
    extent = study_extent(project)
    for x, y, *_ in coordinates_of(display):
        if not in_area(x, y, source_crs):
            raise ValueError("Position outside source CRS area of use")
        if extent:
            west, south, east, north = extent
            if not west <= x <= east or not south <= y <= north:
                warnings.append("Outside project study extent")
        if x == 0 and y == 0:
            warnings.append("Zero/zero coordinate: verify intent")
    properties = {
        key: value for key, value in observation.items() if key != "vertices"
    }
    properties.update({
        "projectId": project["id"],
        "project": project.get("name", ""),
        "sourceCrs": source_id,
        "vertexMetadata": observation["vertices"],
        "observedAt": epoch_iso(observation.get("observedAtEpochMs")),
        "normalizationWarnings": sorted(set(warnings)),
        "transformDescription": operation.description,
        "assets": [
            safe_asset(Path(source_path), reference)
            for reference in observation.get("assets", [])
        ],
    })
    west, south, east, north = display.bounds
    center = display.centroid
    return {
        "id": str(observation["id"]),
        "revision": observation.get("revision", 1),
        "project_id": project["id"],
        "project_name": project.get("name", ""),
        "type": observation.get("type", ""),
        "subtype": observation.get("subtype", ""),
        "observed_at_epoch_ms": observation.get("observedAtEpochMs"),
        "source_crs": source_id,
        "source_crs_wkt": source_crs.to_wkt(),
        "source_wkb": original.wkb,
        "source_vertices_json": json.dumps(observation["vertices"]),
        "properties_json": json.dumps(properties, allow_nan=False),
        "record_hash": digest(observation),
        "transform_description": operation.description,
        "min_lon": west,
        "min_lat": south,
        "max_lon": east,
        "max_lat": north,
        "center_lon": center.x,
        "center_lat": center.y,
        "geometry": display,
    }


def normalize_archive(root=CAPTURES, output=DERIVED):
    latest, _, problems = load_latest(root)
    rows = []
    for item in latest:
        try:
            rows.append(normalize_observation(
                item["observation"], item["project"], item["path"]
            ))
        except Exception as error:
            problems.append({
                "record_id": item["observation"].get("id"),
                "snapshot": str(item["path"]),
                "problem": str(error),
            })
    output.mkdir(parents=True, exist_ok=True)
    features = [{
        "type": "Feature",
        "id": row["id"],
        "geometry": mapping(row["geometry"]),
        "properties": json.loads(row["properties_json"]),
    } for row in rows]
    write_json(output / "observations.geojson", {
        "type": "FeatureCollection", "features": features,
    })
    write_json(output / "review.json", problems)
    write_json(output / "normalization.json", {
        "generatedAt": utc_now(),
        "recordCount": len(rows),
        "problemCount": len(problems),
        "displayCrs": "EPSG:4326",
    })
    if not rows:
        raise ValueError(
            "No valid observations. Review derived/review.json. "
            "No existing Parquet file was replaced."
        )
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    temporary = output / "observations.tmp.parquet"
    frame.to_parquet(temporary, index=False, schema_version="1.0.0")
    temporary.replace(output / "observations.parquet")
    csv_frame = frame.drop(columns=["geometry", "source_wkb"]).copy()
    csv_frame["geometry_wkt"] = frame.geometry.to_wkt()
    csv_frame.to_csv(output / "observations.csv", index=False)
    return frame


def catalog_observations():
    path = DERIVED / "observations.parquet"
    if not path.exists():
        raise FileNotFoundError("Run kwf_geo.py normalize first")
    return pl.scan_parquet(path)


def vertical_unit_factor(reference):
    if reference == "ellipsoidal":
        return 1.0
    crs = resolve_crs(reference)
    if not crs.is_vertical:
        raise ValueError("Expected a vertical CRS")
    factor = crs.axis_info[0].unit_conversion_factor
    if factor is None or factor <= 0:
        raise ValueError("Unknown vertical units")
    return factor


def export_gcps(
    root, output, project_id, target_crs_id, target_vcrs_id,
    site_boundary_path, metric_crs_id, minimum_points=5,
    minimum_hull_area_m2=10.0, minimum_separation_m=1.0,
):
    latest, projects, problems = load_latest(root)
    if problems:
        raise ValueError("Resolve snapshot conflicts/loading errors before GCP export")
    if project_id not in projects:
        raise ValueError("Unknown project")
    project = projects[project_id]
    target = resolve_crs(target_crs_id, project)
    metric = resolve_crs(metric_crs_id, project)
    if not (target.is_geographic or target.is_projected):
        raise ValueError("Target must be a horizontal CRS")
    if not metric.is_projected or any(
        abs(axis.unit_conversion_factor - 1.0) > 1e-12
        for axis in metric.axis_info[:2]
    ):
        raise ValueError("Quality-check CRS must be projected in meters")
    if target_vcrs_id == "unknown":
        raise ValueError("Target height reference cannot be unknown")
    height_factor = vertical_unit_factor(target_vcrs_id)
    boundary_json = read_json(site_boundary_path)
    if boundary_json.get("crs"):
        raise ValueError("Boundary must be RFC 7946 WGS 84 GeoJSON")
    boundary = shape(
        boundary_json["geometry"]
        if boundary_json.get("type") == "Feature"
        else boundary_json
    )
    if (
        boundary.geom_type not in {"Polygon", "MultiPolygon"}
        or not boundary.is_valid or boundary.is_empty
    ):
        raise ValueError("Invalid site boundary")
    bounds = boundary.bounds
    if not (
        -180 <= bounds[0] <= bounds[2] <= 180
        and -90 <= bounds[1] <= bounds[3] <= 90
    ):
        raise ValueError("Boundary must use WGS 84 coordinates")
    candidates = []
    warnings = []
    for item in latest:
        observation = item["observation"]
        if item["project"]["id"] != project_id:
            continue
        if str(observation.get("type", "")).upper() != "GCP":
            continue
        if observation.get("shape") != "Point" or observation.get("surveyed") is not True:
            warnings.append({"id": observation["id"], "reason": "Not a surveyed point"})
            continue
        candidates.append(item)
    if len(candidates) < minimum_points:
        raise ValueError(
            f"Project policy requires at least {minimum_points} surveyed GCPs"
        )
    labels = set()
    exported = []
    metric_points = []
    provenance = []
    for item in candidates:
        observation = item["observation"]
        label = str(observation.get("gcpLabel") or "").strip()
        if not label or label.casefold() in labels:
            raise ValueError("Missing or duplicate GCP label")
        labels.add(label.casefold())
        if not observation.get("surveyReference"):
            raise ValueError(f"{label}: survey reference required")
        if len(observation.get("vertices", [])) != 1:
            raise ValueError(f"{label}: exactly one source vertex required")
        vertex = observation["vertices"][0]
        if vertex.get("source") not in {"manual", "survey"}:
            raise ValueError(f"{label}: invalid survey source")
        if vertex.get("coordinateEpoch") is not None:
            raise ValueError(f"{label}: epoch-dependent transformation not implemented")
        if vertex.get("vcrs") != target_vcrs_id:
            raise ValueError(
                f"{label}: vertical transformation is required but is not "
                "implemented in this release. Do not relabel the height."
            )
        x = finite(vertex.get("x"), f"{label} x")
        y = finite(vertex.get("y"), f"{label} y")
        z = finite(vertex.get("z"), f"{label} z")
        h_accuracy = finite(vertex.get("horizontalAccuracyM"), "Horizontal accuracy")
        v_accuracy = finite(vertex.get("verticalAccuracyM"), "Vertical accuracy")
        if h_accuracy < 0 or v_accuracy < 0:
            raise ValueError("Accuracy cannot be negative")
        source = resolve_crs(vertex["crs"], project)
        if (
            target_vcrs_id == "ellipsoidal"
            and not source.geodetic_crs.equals(target.geodetic_crs)
        ):
            raise ValueError(
                "Ellipsoidal heights across different horizontal datums "
                "require a verified 3D transformation"
            )
        display_operation = transformer(source, CRS.from_epsg(4326))
        longitude, latitude = display_operation.transform(x, y, errcheck=True)
        if not boundary.covers(Point(longitude, latitude)):
            raise ValueError(f"{label}: outside site boundary")
        if not in_area(longitude, latitude, target):
            raise ValueError(f"{label}: outside target CRS area")
        if not in_area(longitude, latitude, metric):
            raise ValueError(f"{label}: outside metric CRS area")
        operation = transformer(source, target)
        tx, ty = operation.transform(x, y, errcheck=True)
        mx, my = transformer(source, metric).transform(x, y, errcheck=True)
        for value in (tx, ty, mx, my):
            finite(value, "Transformed coordinate")
        metric_points.append(Point(mx, my))
        exported.append({
            "name": label, "x": tx, "y": ty, "z": z,
            "h_accuracy": h_accuracy,
            "v_accuracy": v_accuracy / height_factor,
            "longitude": longitude, "latitude": latitude,
            "id": observation["id"],
        })
        provenance.append({
            "id": observation["id"],
            "revision": observation.get("revision", 1),
            "label": label,
            "sourceCrs": vertex["crs"],
            "sourceCoordinates": [x, y, z],
            "sourceHeightReference": vertex["vcrs"],
            "surveyReference": observation["surveyReference"],
            "transformation": operation.description,
            "verticalTransformation": "Identity: matching height references",
        })
    if MultiPoint(metric_points).convex_hull.area < minimum_hull_area_m2:
        raise ValueError("GCP layout fails configured hull-area threshold")
    for index, point in enumerate(metric_points):
        if any(point.distance(other) < minimum_separation_m
               for other in metric_points[index + 1:]):
            raise ValueError("GCPs fail configured separation threshold")
    export_id = str(uuid.uuid4())
    directory = output / export_id
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "gcp_ground.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "x", "y", "z", "h_accuracy", "v_accuracy"])
        precision = 9 if target.is_geographic else 3
        for point in exported:
            writer.writerow([
                point["name"], f'{point["x"]:.{precision}f}',
                f'{point["y"]:.{precision}f}', f'{point["z"]:.3f}',
                f'{point["h_accuracy"]:.3f}', f'{point["v_accuracy"]:.3f}',
            ])
    write_json(directory / "gcp_points.geojson", {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "id": point["id"],
            "geometry": {
                "type": "Point",
                "coordinates": [point["longitude"], point["latitude"]],
            },
            "properties": {"name": point["name"]},
        } for point in exported],
    })
    write_json(directory / "gcp_export.json", {
        "exportId": export_id,
        "projectId": project_id,
        "exportedAt": utc_now(),
        "targetCrs": target_crs_id,
        "targetCrsWkt": target.to_wkt(),
        "targetHeightReference": target_vcrs_id,
        "horizontalAccuracyUnit": "meter",
        "verticalAccuracyUnit": (
            "meter" if target_vcrs_id == "ellipsoidal"
            else resolve_crs(target_vcrs_id).axis_info[0].unit_name
        ),
        "points": provenance,
        "warnings": warnings,
        "imagePixelCorrespondencesIncluded": False,
        "verticalConversionSupported": False,
    })
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
    args = parser.parse_args()
    if args.command == "normalize":
        frame = normalize_archive(args.captures, args.output)
        print(f"Normalized {len(frame)} observations")
    else:
        result = export_gcps(
            args.captures, args.output, args.project_id, args.target_crs,
            args.target_vcrs, args.site_boundary, args.metric_crs,
        )
        print(result)


if __name__ == "__main__":
    main()

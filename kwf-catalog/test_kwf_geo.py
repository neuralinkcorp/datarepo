import csv
import json

import geopandas as gpd
import pyarrow.parquet as pq
import pytest

from kwf_geo import (
    export_gcps,
    load_latest,
    normalize_archive,
    normalize_observation,
)


def project(project_id="project-test"):
    return {
        "id": project_id,
        "name": "Arkansas Test",
        "studyExtent": [-95, 33, -89, 37],
    }


def observation(project_id="project-test"):
    return {
        "id": "record-test",
        "revision": 1,
        "projectId": project_id,
        "type": "Observation",
        "subtype": "Vegetation",
        "shape": "Point",
        "observedAtEpochMs": 1700000000000,
        "vertices": [{
            "x": -94.1,
            "y": 35.1,
            "crs": "EPSG:4326",
            "vcrs": "unknown",
            "source": "manual",
        }],
        "assets": [],
    }


def snapshot(root, batch, record, project_info=None):
    project_info = project(record["projectId"]) if project_info is None else project_info
    directory = root / record["projectId"] / batch
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "source-observations.json"
    path.write_text(json.dumps({
        "formatVersion": 2,
        "batchId": batch,
        "project": project_info,
        "observations": [record],
    }), encoding="utf-8")
    return path


def gcps(count=5, project_id="project-test"):
    points = [
        (-94.1000, 35.1000),
        (-94.0990, 35.1000),
        (-94.1000, 35.1010),
        (-94.0990, 35.1010),
        (-94.0995, 35.1005),
    ][:count]
    return [{
        "id": f"gcp-{index}",
        "revision": 1,
        "projectId": project_id,
        "type": "GCP",
        "shape": "Point",
        "surveyed": True,
        "gcpLabel": f"CP-{index}",
        "surveyReference": "survey-2026",
        "vertices": [{
            "x": x,
            "y": y,
            "z": 100.0,
            "crs": "EPSG:4326",
            "vcrs": "ellipsoidal",
            "source": "survey",
            "horizontalAccuracyM": 0.02,
            "verticalAccuracyM": 0.03,
        }],
    } for index, (x, y) in enumerate(points)]


def write_boundary(path, bounds=(-95, 33, -89, 37)):
    west, south, east, north = bounds
    path.write_text(json.dumps({
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south], [east, north],
            [west, north], [west, south],
        ]],
    }), encoding="utf-8")
    return path


def test_preserves_original_geometry(tmp_path):
    row = normalize_observation(
        observation(), project(), tmp_path / "source-observations.json"
    )
    assert row["source_wkb"]
    assert row["source_crs"] == "EPSG:4326"
    assert row["geometry"].x == pytest.approx(-94.1)


def test_transforms_projected_coordinates(tmp_path):
    record = observation()
    record["vertices"] = [{
        "x": 500000,
        "y": 3880000,
        "crs": "EPSG:32615",
        "source": "manual",
    }]
    row = normalize_observation(
        record, project(), tmp_path / "source-observations.json"
    )
    assert row["geometry"].x == pytest.approx(-93, abs=0.001)
    assert row["source_crs"] == "EPSG:32615"
    assert row["source_wkb"] != row["geometry"].wkb


def test_rejects_text_coordinates(tmp_path):
    record = observation()
    record["vertices"][0]["x"] = "94 W"
    with pytest.raises(ValueError):
        normalize_observation(
            record, project(), tmp_path / "source-observations.json"
        )


def test_conflicts_are_reported(tmp_path):
    root = tmp_path / "captures"
    snapshot(root, "a", observation())
    changed = observation()
    changed["notes"] = "Conflicting same revision"
    snapshot(root, "b", changed)
    _, _, problems = load_latest(root)
    assert any("Conflicting" in item["problem"] for item in problems)


def test_latest_revision_wins(tmp_path):
    root = tmp_path / "captures"
    snapshot(root, "a", observation())
    changed = observation()
    changed["revision"] = 2
    snapshot(root, "b", changed)
    rows, _, problems = load_latest(root)
    assert not problems
    assert rows[0]["revision"] == 2


def test_real_geoparquet_metadata(tmp_path):
    root = tmp_path / "captures"
    snapshot(root, "a", observation())
    output = tmp_path / "derived"
    normalize_archive(root, output)
    path = output / "observations.parquet"
    metadata = pq.read_metadata(path).metadata
    geo = json.loads(metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    frame = gpd.read_parquet(path)
    assert frame.crs.to_epsg() == 4326
    assert "source_crs" in frame
    assert "source_wkb" in frame


def test_source_timestamp_and_asset_reference_survive(tmp_path):
    root = tmp_path / "captures"
    batch = root / "project-test" / "batch-a"
    assets = batch / "assets"
    assets.mkdir(parents=True)
    (assets / "sample.jpg").write_bytes(b"sample")
    record = observation()
    record["assets"] = ["assets/sample.jpg"]
    snapshot(root, "batch-a", record)
    output = tmp_path / "derived"
    normalize_archive(root, output)
    data = json.loads((output / "observations.geojson").read_text())
    properties = data["features"][0]["properties"]
    assert properties["observedAt"] is not None
    assert properties["assets"] == [
        "captures/project-test/batch-a/assets/sample.jpg"
    ]


def test_invalid_polygon_is_reported_not_written(tmp_path):
    record = observation()
    record["shape"] = "Polygon"
    record["vertices"] = [
        {"x": -94.1, "y": 35.1, "crs": "EPSG:4326"},
        {"x": -94.0, "y": 35.2, "crs": "EPSG:4326"},
        {"x": -94.1, "y": 35.2, "crs": "EPSG:4326"},
        {"x": -94.0, "y": 35.1, "crs": "EPSG:4326"},
    ]
    root = tmp_path / "captures"
    snapshot(root, "a", record)
    with pytest.raises(ValueError, match="No valid observations"):
        normalize_archive(root, tmp_path / "derived")
    review = json.loads((tmp_path / "derived" / "review.json").read_text())
    assert any("Invalid" in problem["problem"] for problem in review)


def test_gcp_export_rejects_records_from_other_project(tmp_path):
    root = tmp_path / "captures"
    for index, record in enumerate(gcps(4)):
        record["id"] = f"a-{index}"
        snapshot(root, f"a-{index}", record)
    other_project = "other-project"
    other = gcps(1, other_project)[0]
    snapshot(root, "other-1", other, project(other_project))
    with pytest.raises(ValueError, match="at least 5"):
        export_gcps(
            root, tmp_path / "exports", "project-test", "EPSG:4326",
            "ellipsoidal", write_boundary(tmp_path / "boundary.json"),
            "EPSG:32615",
        )


def test_gcp_export_rejects_duplicate_labels(tmp_path):
    root = tmp_path / "captures"
    records = gcps()
    records[1]["gcpLabel"] = records[0]["gcpLabel"]
    for index, record in enumerate(records):
        snapshot(root, str(index), record)
    with pytest.raises(ValueError, match="duplicate GCP label"):
        export_gcps(
            root, tmp_path / "exports", "project-test", "EPSG:4326",
            "ellipsoidal", write_boundary(tmp_path / "boundary.json"),
            "EPSG:32615",
        )


def test_gcp_export_rejects_boundary_violation(tmp_path):
    root = tmp_path / "captures"
    records = gcps()
    records[0]["vertices"][0]["x"] = -95.0
    for index, record in enumerate(records):
        snapshot(root, str(index), record)
    with pytest.raises(ValueError, match="outside site boundary"):
        export_gcps(
            root, tmp_path / "exports", "project-test", "EPSG:4326",
            "ellipsoidal", write_boundary(
                tmp_path / "boundary.json", (-94.5, 34.5, -93.5, 35.5)
            ), "EPSG:32615",
        )


def test_gcp_export_rejects_insufficient_points(tmp_path):
    root = tmp_path / "captures"
    for index, record in enumerate(gcps(4)):
        snapshot(root, str(index), record)
    with pytest.raises(ValueError, match="at least 5"):
        export_gcps(
            root, tmp_path / "exports", "project-test", "EPSG:4326",
            "ellipsoidal", write_boundary(tmp_path / "boundary.json"),
            "EPSG:32615",
        )


def test_gcp_export_rejects_close_points(tmp_path):
    root = tmp_path / "captures"
    records = gcps()
    records[1]["vertices"][0]["x"] = records[0]["vertices"][0]["x"] + 0.000001
    records[1]["vertices"][0]["y"] = records[0]["vertices"][0]["y"]
    for index, record in enumerate(records):
        snapshot(root, str(index), record)
    with pytest.raises(ValueError, match="separation"):
        export_gcps(
            root, tmp_path / "exports", "project-test", "EPSG:4326",
            "ellipsoidal", write_boundary(tmp_path / "boundary.json"),
            "EPSG:32615",
        )


def test_gcp_export_rejects_unimplemented_vertical_conversion(tmp_path):
    root = tmp_path / "captures"
    records = gcps()
    records[0]["vertices"][0]["vcrs"] = "EPSG:5703"
    for index, record in enumerate(records):
        snapshot(root, str(index), record)
    with pytest.raises(ValueError, match="vertical transformation"):
        export_gcps(
            root, tmp_path / "exports", "project-test", "EPSG:4326",
            "ellipsoidal", write_boundary(tmp_path / "boundary.json"),
            "EPSG:32615",
        )


def test_gcp_export_writes_ground_coordinates_and_references(tmp_path):
    root = tmp_path / "captures"
    for index, record in enumerate(gcps()):
        snapshot(root, str(index), record)
    directory = export_gcps(
        root, tmp_path / "exports", "project-test", "EPSG:32615",
        "ellipsoidal", write_boundary(tmp_path / "boundary.json"),
        "EPSG:32615",
    )
    with (directory / "gcp_ground.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert 100000 < float(rows[0]["x"]) < 900000
    assert float(rows[0]["y"]) > 3_000_000
    metadata = json.loads((directory / "gcp_export.json").read_text())
    assert metadata["targetCrs"] == "EPSG:32615"
    assert metadata["targetHeightReference"] == "ellipsoidal"
    assert metadata["points"][0]["sourceCoordinates"][2] == 100.0

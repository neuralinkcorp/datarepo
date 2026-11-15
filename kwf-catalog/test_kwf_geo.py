import json
import pytest
import geopandas as gpd
import pyarrow.parquet as pq
from pathlib import Path

from kwf_geo import normalize_observation, normalize_archive, load_latest


def project():
    return {"id": "proj-001", "name": "Test Project"}


def observation(obs_id="obs-001"):
    return {
        "id": obs_id,
        "revision": 1,
        "type": "Vegetation",
        "subtype": "Tree",
        "shape": "Point",
        "vertices": [{"x": -94.0, "y": 35.0, "crs": "EPSG:4326"}],
        "notes": "Test note",
        "surveyed": False,
    }


def write_snapshot(root: Path, batch_id: str, obs: dict, proj=None):
    if proj is None:
        proj = project()
    path = root / "proj-001" / batch_id / "source-observations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "project": proj,
        "observations": [obs],
    }), encoding="utf-8")
    return path


def test_point_observation_has_geometry(tmp_path):
    obs = observation()
    row = normalize_observation(obs, project(), tmp_path / "source-observations.json")
    assert row["geometry"] is not None
    assert row["geometry"].geom_type == "Point"
    assert row["pending_normalization"] is False


def test_non_wgs84_vertex_is_pending(tmp_path):
    obs = observation()
    obs["vertices"] = [{"x": 500000.0, "y": 4000000.0, "crs": "EPSG:32614"}]
    row = normalize_observation(obs, project(), tmp_path / "source-observations.json")
    assert row["geometry"] is None
    assert row["pending_normalization"] is True


def test_conflicting_revision_is_not_silently_selected(tmp_path):
    root = tmp_path / "captures"
    first = observation()
    write_snapshot(root, "batch-a", first)

    second = observation()
    second["notes"] = "Different content with the same revision"
    write_snapshot(root, "batch-b", second)

    with pytest.raises(ValueError, match="Conflicting"):
        load_latest(root)


def test_latest_revision_wins(tmp_path):
    root = tmp_path / "captures"
    write_snapshot(root, "batch-a", observation())

    changed = observation()
    changed["revision"] = 2
    changed["notes"] = "Updated"
    write_snapshot(root, "batch-b", changed)

    rows, _ = load_latest(root)

    assert len(rows) == 1
    assert rows[0][0]["revision"] == 2


def test_geoparquet_has_crs_and_geometry_metadata(tmp_path):
    root = tmp_path / "captures"
    write_snapshot(root, "batch-a", observation())

    output = tmp_path / "derived"
    normalize_archive(root, output)

    path = output / "observations.parquet"
    metadata = pq.read_metadata(path).metadata

    assert b"geo" in metadata

    geo = json.loads(metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"

    frame = gpd.read_parquet(path)
    assert frame.crs.to_epsg() == 4326
    assert "source_wkb" in frame.columns


def test_polygon_preserves_full_geometry(tmp_path):
    record = observation()
    record["shape"] = "Polygon"
    record["vertices"] = [
        {"x": -94.1, "y": 35.1, "crs": "EPSG:4326"},
        {"x": -94.0, "y": 35.1, "crs": "EPSG:4326"},
        {"x": -94.0, "y": 35.2, "crs": "EPSG:4326"},
    ]

    row = normalize_observation(
        record, project(), tmp_path / "source-observations.json"
    )

    assert row["geometry"].geom_type == "Polygon"
    assert len(row["geometry"].exterior.coords) == 4
    assert row["max_lat"] == pytest.approx(35.2)

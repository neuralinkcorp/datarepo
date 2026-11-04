from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "kwf-field-data" / "captures"


def observations_frame(root: Path = CAPTURES) -> pl.LazyFrame:
    latest: dict[str, dict] = {}

    for source in sorted(root.glob("*/*/observations.geojson")):
        document = json.loads(source.read_text(encoding="utf-8"))

        if document.get("type") != "FeatureCollection":
            raise ValueError(f"Invalid FeatureCollection: {source}")

        for feature in document.get("features", []):
            record_id = str(feature["id"])
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            latitude = None
            longitude = None

            if geometry.get("type") == "Point":
                coordinates = geometry.get("coordinates", [])

                if len(coordinates) >= 2:
                    longitude, latitude = coordinates[:2]

            row = {
                "id": record_id,
                "project": properties.get("project"),
                "type": properties.get("type"),
                "subtype": properties.get("subtype"),
                "captured_at_epoch_ms": properties.get("capturedAtEpochMs"),
                "geometry_type": geometry.get("type"),
                "longitude": longitude,
                "latitude": latitude,
                "geometry_json": json.dumps(geometry),
                "soil_ph": properties.get("soilPh"),
                "moisture": properties.get("moisture"),
                "notes": properties.get("notes"),
                "surveyed": properties.get("surveyed", False),
                "survey_reference": properties.get("surveyReference"),
                "elevation_meters": properties.get("elevationMeters"),
                "vertical_datum": properties.get("verticalDatum"),
                "vertex_metadata_json": json.dumps(
                    properties.get("vertexMetadata", [])
                ),
                "assets_json": json.dumps(properties.get("assets", [])),
                "snapshot_directory": str(source.parent.relative_to(ROOT)),
            }

            previous = latest.get(record_id)

            if (
                previous is None
                or (row["captured_at_epoch_ms"] or 0)
                >= (previous["captured_at_epoch_ms"] or 0)
            ):
                latest[record_id] = row

    schema = {
        "id": pl.String,
        "project": pl.String,
        "type": pl.String,
        "subtype": pl.String,
        "captured_at_epoch_ms": pl.Int64,
        "geometry_type": pl.String,
        "longitude": pl.Float64,
        "latitude": pl.Float64,
        "geometry_json": pl.String,
        "soil_ph": pl.Float64,
        "moisture": pl.Float64,
        "notes": pl.String,
        "surveyed": pl.Boolean,
        "survey_reference": pl.String,
        "elevation_meters": pl.Float64,
        "vertical_datum": pl.String,
        "vertex_metadata_json": pl.String,
        "assets_json": pl.String,
        "snapshot_directory": pl.String,
    }

    return pl.DataFrame(
        list(latest.values()),
        schema=schema,
        strict=False,
    ).lazy()


def read_dataset(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pl.scan_parquet(path)

    if suffix == ".csv":
        return pl.scan_csv(path)

    if suffix in {".geojson", ".json"}:
        document = json.loads(path.read_text(encoding="utf-8"))

        rows = [
            {
                "id": str(feature["id"]),
                "properties_json": json.dumps(
                    feature.get("properties", {})
                ),
                "geometry_json": json.dumps(
                    feature.get("geometry", {})
                ),
            }
            for feature in document["features"]
        ]

        return pl.DataFrame(
            rows,
            schema={
                "id": pl.String,
                "properties_json": pl.String,
                "geometry_json": pl.String,
            },
        ).lazy()

    raise ValueError(f"Unsupported file format: {suffix}")


# Optional registration with the existing upstream datarepo library.
# The adapter remains executable even when that library isn't installed.
try:
    from datarepo.core import table

    @table(
        data_input="KWF immutable field snapshots",
        latency_info="Updated after repository snapshots are pulled",
    )
    def observations() -> pl.LazyFrame:
        """KWF field observations with stable-ID deduplication."""
        return observations_frame()

except ImportError:
    def observations() -> pl.LazyFrame:
        return observations_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "kwf-field-data" / "derived",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    frame = observations_frame().collect()

    frame.write_parquet(args.output / "observations.parquet")
    frame.write_csv(args.output / "observations.csv")

    print(f"Exported {frame.height} unique observations.")

from __future__ import annotations

import json
import shutil

from kwf_geo import DATA_ROOT, DERIVED, ROOT, normalize_archive


PUBLIC = ROOT / "habitatmapper" / "public" / "field-data"


def main():
    normalize_archive()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for filename in (
        "observations.geojson",
        "review.json",
        "normalization.json",
    ):
        shutil.copy2(DERIVED / filename, PUBLIC / filename)

    document = json.loads(
        (DERIVED / "observations.geojson").read_text(encoding="utf-8")
    )
    for feature in document["features"]:
        for reference in feature["properties"].get("assets", []):
            source = (DATA_ROOT / reference).resolve()
            try:
                source.relative_to(DATA_ROOT.resolve())
            except ValueError as error:
                raise ValueError("Asset leaves the dataset directory") from error
            if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not source.is_file():
                continue
            destination = (PUBLIC / reference).resolve()
            try:
                destination.relative_to(PUBLIC.resolve())
            except ValueError as error:
                raise ValueError("Asset destination leaves viewer data") from error
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print(f"Published local viewer data to {PUBLIC}")


if __name__ == "__main__":
    main()

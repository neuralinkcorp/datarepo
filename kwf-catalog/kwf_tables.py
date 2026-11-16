from __future__ import annotations

from kwf_geo import CAPTURES, DERIVED, catalog_observations, normalize_archive


def observations_frame():
    return catalog_observations()


if __name__ == "__main__":
    frame = normalize_archive(CAPTURES, DERIVED)
    print(f"Normalized {len(frame)} observations")

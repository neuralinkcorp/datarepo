from __future__ import annotations
import sys
from kwf_geo import catalog_observations

try:
    from datarepo.core import Catalog, ModuleDatabase, table

    @table(
        data_input="KWF normalized field observations in GeoParquet",
        latency_info="Updated after snapshot normalization",
    )
    def observations():
        """Field observations with original and WGS 84 geometry."""
        return catalog_observations()

    KWFCatalog = Catalog({
        "field": ModuleDatabase(sys.modules[__name__])
    })
except (ImportError, ModuleNotFoundError):
    def observations():
        """Field observations with original and WGS 84 geometry."""
        return catalog_observations()

    KWFCatalog = None

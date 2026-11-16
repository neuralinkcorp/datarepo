from __future__ import annotations

import sys

from datarepo.core import Catalog, ModuleDatabase, table

from kwf_geo import catalog_observations


@table(
    data_input="KWF normalized GeoParquet observations",
    latency_info="Updated after snapshot normalization",
)
def observations():
    """Queryable field records; geometry columns are WKB."""
    return catalog_observations()


KWFCatalog = Catalog({
    "field": ModuleDatabase(sys.modules[__name__])
})

# Python compatibility

The minimum Python version is 3.10. Runtime type aliases use the union operator,
and Python 3.9 fails while importing `datarepo.core.tables.filters` even after
its dependencies install. The existing Python 3.8 declaration also cannot
resolve a binary-wheel environment for the required Delta Lake versions on
the tested Linux x86-64 host. These are distinct failures.

The installed-wheel smoke workflow builds the package, including its static
catalog, then installs into a fresh environment and reads a local Parquet table
with Python 3.10 and 3.12 on Ubuntu 24.04. It runs with `python -I` outside the
source directory and verifies the package comes from `site-packages`.

Building from source needs Python build tooling, Node.js and npm because the
existing Hatch hook compiles the web catalog. The dependency caps for Polars
and Delta Lake are unchanged. This check covers installation, import, and one
local table read; it does not establish every backend or platform combination.
Other Python versions, Windows and macOS are not validated by this Linux job.

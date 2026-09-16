import json
from pathlib import Path
import runpy

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "local_quickstart.py"


def test_local_quickstart_schema_filter_and_join():
    result = runpy.run_path(str(EXAMPLE))["run_example"]()
    assert result.schema == {
        "part_id": pl.Int64,
        "part_name": pl.String,
        "supplier_id": pl.Int64,
        "supplier_name": pl.String,
    }
    assert result.to_dicts() == [
        {
            "part_id": 1,
            "part_name": "Bolt",
            "supplier_id": 20,
            "supplier_name": "Supplier B",
        },
        {
            "part_id": 2,
            "part_name": "Nut",
            "supplier_id": 10,
            "supplier_name": "Supplier A",
        },
        {
            "part_id": 3,
            "part_name": "Washer",
            "supplier_id": 20,
            "supplier_name": "Supplier B",
        },
    ]


def test_readme_code_and_output_match_executable_example(capsys):
    readme = (ROOT / "docs" / "README.md").read_text()
    code = readme.split("<!-- local-quickstart-code -->\n```python\n", 1)[1].split(
        "```", 1
    )[0]
    assert code == EXAMPLE.read_text()
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    actual = json.loads(capsys.readouterr().out)
    documented = readme.split("<!-- local-quickstart-output -->\n```json\n", 1)[
        1
    ].split("```", 1)[0]
    assert actual == json.loads(documented)

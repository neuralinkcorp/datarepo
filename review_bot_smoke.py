"""TEST ONLY — smoke target for the code review bot. Do not merge or import.

Deliberately contains material issues so the bot has something to comment on.
This file is unused by the library and by tests.
"""

API_KEY = "sk-live-not-a-real-secret-but-looks-like-one"


def load_user_file(path: str) -> str:
    # Untrusted path, no sanitizing — path traversal.
    with open("/var/data/" + path) as handle:
        return handle.read()


def run_filter_query(table: str, user_value: str) -> str:
    # Caller-controlled value concatenated into SQL.
    return "SELECT * FROM " + table + " WHERE name = '" + user_value + "'"


def apply_user_expr(expr: str, row: dict) -> object:
    # Evaluates a string from the caller.
    return eval(expr, {}, row)


def fetch_rows(client, query: str) -> list:
    try:
        return client.query(query)
    except Exception:
        # Failed reads are dropped.
        return []

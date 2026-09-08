import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "grok_review.py"
WORKFLOW = ROOT / ".github" / "workflows" / "grok-review.yml"


def load_grok_review():
    spec = importlib.util.spec_from_file_location("grok_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grok():
    return load_grok_review()


SAMPLE_PATCH = """\
@@ -10,7 +10,9 @@ def load_rows(filters):
     query = build_query(filters)
-    return client.query(query)
+    if filters is None:
+        filters = []
+    return client.query(query, filters)
"""


class FakeGitHub:
    def __init__(self, routes):
        self.repo = "neuralinkcorp/datarepo"
        self.routes = routes
        self.posts = []

    def _lookup(self, path, params=None):
        key = path
        if params:
            encoded = "&".join(f"{k}={params[k]}" for k in sorted(params))
            key = f"{path}?{encoded}"
        if key not in self.routes:
            raise KeyError(key)
        return self.routes[key]

    def get_json(self, path, params=None):
        return self._lookup(path, params)

    def get_all(self, path, params=None):
        value = self._lookup(path, params)
        return list(value)

    def post_json(self, path, payload):
        self.posts.append((path, payload))
        return 200, {"id": 99, "event": payload.get("event")}


def test_workflow_uses_workflow_run_not_pull_request_target():
    text = WORKFLOW.read_text()
    on_block = text.split("\njobs:", 1)[0]
    assert "workflow_run:" in on_block
    assert "pull_request_target:" not in on_block
    assert "Test, Build and Publish datarepo" in text
    assert "types:\n      - requested" in text
    assert "persist-credentials: false" in text


def test_right_side_lines_include_added_and_context_not_deleted(grok):
    lines = grok.right_side_lines(SAMPLE_PATCH)
    assert lines == {10, 11, 12, 13}
    assert 9 not in lines


def test_right_side_lines_empty_and_no_newline(grok):
    assert grok.right_side_lines(None) == set()
    patch = "@@ -1 +1 @@\n-old\n+new\n\\ No newline at end of file\n"
    assert grok.right_side_lines(patch) == {1}


def test_select_inline_comments_keeps_valid_high_severity_first(grok):
    valid = {"src/datarepo/core/tables/clickhouse_table.py": {12, 40, 41}}
    raw = [
        {
            "path": "src/datarepo/core/tables/clickhouse_table.py",
            "line": 40,
            "severity": "low",
            "body": "nit",
        },
        {
            "path": "src/datarepo/core/tables/clickhouse_table.py",
            "line": 12,
            "severity": "high",
            "body": "null filter dropped",
        },
        {
            "path": "src/datarepo/core/tables/clickhouse_table.py",
            "line": 99,
            "severity": "high",
            "body": "not in the diff",
        },
        {
            "path": "missing.py",
            "line": 1,
            "severity": "high",
            "body": "unknown file",
        },
        {
            "path": "src/datarepo/core/tables/clickhouse_table.py",
            "line": 12,
            "severity": "medium",
            "body": "duplicate line",
        },
    ]
    selected = grok.select_inline_comments(raw, valid)
    assert [c["line"] for c in selected] == [12, 40]
    assert selected[0]["severity"] == "high"


def test_select_inline_comments_caps_at_eight(grok):
    valid = {"a.py": set(range(1, 20))}
    raw = [
        {"path": "a.py", "line": i, "severity": "medium", "body": f"c{i}"}
        for i in range(1, 16)
    ]
    selected = grok.select_inline_comments(raw, valid)
    assert len(selected) == 8


def test_parse_model_output_strips_fences(grok):
    text = """```json
{"summary": "Looks fine.", "comments": []}
```"""
    parsed = grok.parse_model_output(text)
    assert parsed["summary"] == "Looks fine."
    assert parsed["comments"] == []


def test_build_review_payload_is_always_comment(grok):
    payload = grok.build_review_payload(
        "abc123",
        "summary",
        [{"path": "a.py", "line": 1, "body": "bug"}],
    )
    assert payload["event"] == "COMMENT"
    assert payload["event"] not in {"APPROVE", "REQUEST_CHANGES"}
    assert payload["comments"][0]["side"] == "RIGHT"
    assert "not a maintainer approval" in payload["body"]


def test_resolve_pull_number_uses_head_selector_when_payload_empty(grok):
    event = {
        "workflow_run": {
            "event": "pull_request",
            "head_sha": "deadbeef",
            "head_branch": "fix-null",
            "head_repository": {
                "full_name": "zack-dev-cm/datarepo",
                "owner": {"login": "zack-dev-cm"},
            },
            "pull_requests": [],
        }
    }
    github = FakeGitHub(
        {
            "/repos/neuralinkcorp/datarepo/commits/deadbeef/pulls": [],
            "/repos/neuralinkcorp/datarepo/pulls?head=zack-dev-cm:fix-null&state=open": [
                {"number": 57, "state": "open"}
            ],
        }
    )
    assert grok.resolve_pull_number(github, event) == 57


def test_skip_draft_and_existing_review(grok):
    event = {"workflow_run": {"event": "pull_request"}}
    draft = grok.skip_reason(
        event=event,
        pr={"draft": True, "user": {"login": "alice"}},
        bot_login="neuralink-code-review-bot[bot]",
        reviews=[],
        head_sha="aaa",
    )
    assert draft and "draft" in draft

    already = grok.skip_reason(
        event=event,
        pr={"draft": False, "user": {"login": "alice"}},
        bot_login="neuralink-code-review-bot[bot]",
        reviews=[
            {
                "user": {"login": "neuralink-code-review-bot[bot]"},
                "commit_id": "aaa",
                "state": "COMMENTED",
            }
        ],
        head_sha="aaa",
    )
    assert already and "already reviewed" in already

    own = grok.skip_reason(
        event=event,
        pr={"draft": False, "user": {"login": "neuralink-code-review-bot[bot]"}},
        bot_login="neuralink-code-review-bot[bot]",
        reviews=[],
        head_sha="aaa",
    )
    assert own and "review bot" in own


def test_run_skips_without_secrets(grok):
    message = grok.run({}, {}, FakeGitHub({}), lambda *args: "")
    assert message.startswith("skip:")


def test_run_posts_comment_review(grok):
    event = {
        "workflow_run": {
            "event": "pull_request",
            "head_sha": "abc",
            "pull_requests": [{"number": 12}],
        }
    }
    github = FakeGitHub(
        {
            "/repos/neuralinkcorp/datarepo/pulls/12": {
                "number": 12,
                "title": "Fix nulls",
                "body": "Handle IS NULL",
                "draft": False,
                "user": {"login": "zack-dev-cm"},
                "head": {"sha": "abc"},
            },
            "/user": {"login": "neuralink-code-review-bot[bot]"},
            "/repos/neuralinkcorp/datarepo/pulls/12/reviews": [],
            "/repos/neuralinkcorp/datarepo/pulls/12/files": [
                {
                    "filename": "src/datarepo/core/tables/clickhouse_table.py",
                    "status": "modified",
                    "patch": SAMPLE_PATCH,
                }
            ],
        }
    )

    def complete(api_key, model, system, user):
        assert api_key == "xai-test"
        assert "clickhouse_table.py" in user
        return json.dumps(
            {
                "summary": "Null filters need coverage.",
                "comments": [
                    {
                        "path": "src/datarepo/core/tables/clickhouse_table.py",
                        "line": 12,
                        "severity": "high",
                        "body": "filters is rebound after query is built",
                    }
                ],
            }
        )

    message = grok.run(
        {"GITHUB_TOKEN": "t", "XAI_API_KEY": "xai-test"},
        event,
        github,
        complete,
    )
    assert "posted review 99" in message
    assert len(github.posts) == 1
    payload = github.posts[0][1]
    assert payload["event"] == "COMMENT"
    assert payload["commit_id"] == "abc"
    assert payload["comments"][0]["line"] == 12


def test_format_diff_omits_binary_and_respects_cap(grok):
    files = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b\n"},
        {"filename": "photo.png", "status": "added"},
    ]
    text, omitted = grok.format_diff(files)
    assert "a.py" in text
    assert "photo.png" in omitted
    _, omitted_small = grok.format_diff(files, limit=10)
    assert "a.py" in omitted_small


def test_http_json_not_used_for_pr_text_in_shell(grok):
    # Guardrail: the reviewer must talk to GitHub/xAI via urllib, not os.system.
    source = SCRIPT.read_text()
    assert "os.system" not in source
    assert "subprocess" not in source
    assert "pull_request_target" not in source
    assert 'REVIEW_EVENT = "COMMENT"' in source


def test_parse_next_link(grok):
    header = (
        '<https://api.github.com/repos/o/r/pulls/1/files?page=2>; rel="next", '
        '<https://api.github.com/repos/o/r/pulls/1/files?page=3>; rel="last"'
    )
    assert grok.parse_next_link(header).endswith("page=2")
    assert grok.parse_next_link(None) is None

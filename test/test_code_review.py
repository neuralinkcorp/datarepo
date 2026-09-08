import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "code_review.py"
WORKFLOW = ROOT / ".github" / "workflows" / "code-review.yml"


def load_code_review():
    spec = importlib.util.spec_from_file_location("code_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reviewer():
    return load_code_review()


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


def test_right_side_lines_include_added_and_context_not_deleted(reviewer):
    lines = reviewer.right_side_lines(SAMPLE_PATCH)
    assert lines == {10, 11, 12, 13}
    assert 9 not in lines


def test_right_side_lines_empty_and_no_newline(reviewer):
    assert reviewer.right_side_lines(None) == set()
    patch = "@@ -1 +1 @@\n-old\n+new\n\\ No newline at end of file\n"
    assert reviewer.right_side_lines(patch) == {1}


def test_select_inline_comments_keeps_valid_high_severity_first(reviewer):
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
    selected = reviewer.select_inline_comments(raw, valid)
    assert [c["line"] for c in selected] == [12, 40]
    assert selected[0]["severity"] == "high"


def test_select_inline_comments_caps_at_eight(reviewer):
    valid = {"a.py": set(range(1, 20))}
    raw = [
        {"path": "a.py", "line": i, "severity": "medium", "body": f"c{i}"}
        for i in range(1, 16)
    ]
    selected = reviewer.select_inline_comments(raw, valid)
    assert len(selected) == 8


def test_parse_model_output_strips_fences(reviewer):
    text = """```json
{"summary": "Looks fine.", "comments": []}
```"""
    parsed = reviewer.parse_model_output(text)
    assert parsed["summary"] == "Looks fine."
    assert parsed["comments"] == []


def test_build_review_payload_is_always_comment(reviewer):
    payload = reviewer.build_review_payload(
        "abc123",
        "summary",
        [{"path": "a.py", "line": 1, "body": "bug"}],
    )
    assert payload["event"] == "COMMENT"
    assert payload["event"] not in {"APPROVE", "REQUEST_CHANGES"}
    assert payload["comments"][0]["side"] == "RIGHT"
    assert "not a maintainer approval" in payload["body"]


def test_resolve_pull_number_uses_head_selector_when_payload_empty(reviewer):
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
    assert reviewer.resolve_pull_number(github, event) == 57


def test_skip_draft_and_existing_review(reviewer):
    event = {"workflow_run": {"event": "pull_request"}}
    draft = reviewer.skip_reason(
        event=event,
        pr={"draft": True, "user": {"login": "alice"}},
        bot_login="code-review-bot[bot]",
        reviews=[],
        head_sha="aaa",
    )
    assert draft and "draft" in draft

    already = reviewer.skip_reason(
        event=event,
        pr={"draft": False, "user": {"login": "alice"}},
        bot_login="code-review-bot[bot]",
        reviews=[
            {
                "user": {"login": "code-review-bot[bot]"},
                "commit_id": "aaa",
                "state": "COMMENTED",
            }
        ],
        head_sha="aaa",
    )
    assert already and "already reviewed" in already

    own = reviewer.skip_reason(
        event=event,
        pr={"draft": False, "user": {"login": "code-review-bot[bot]"}},
        bot_login="code-review-bot[bot]",
        reviews=[],
        head_sha="aaa",
    )
    assert own and "review bot" in own


def test_run_skips_without_secrets(reviewer):
    message = reviewer.run({}, {}, FakeGitHub({}), lambda *args: "")
    assert message.startswith("skip:")


def test_run_posts_comment_review(reviewer):
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
            "/user": {"login": "code-review-bot[bot]"},
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
        assert api_key == "test-key"
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

    message = reviewer.run(
        {
            "GITHUB_TOKEN": "t",
            "MODEL_API_KEY": "test-key",
            "MODEL": "test-model",
            "MODEL_API_URL": "https://example.invalid/v1/chat/completions",
        },
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


def test_format_diff_omits_binary_and_respects_cap(reviewer):
    files = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b\n"},
        {"filename": "photo.png", "status": "added"},
    ]
    text, omitted = reviewer.format_diff(files)
    assert "a.py" in text
    assert "photo.png" in omitted
    _, omitted_small = reviewer.format_diff(files, limit=10)
    assert "a.py" in omitted_small


def test_http_json_not_used_for_pr_text_in_shell(reviewer):
    # Guardrail: the reviewer must talk to the model API via urllib, not os.system.
    source = SCRIPT.read_text()
    assert "os.system" not in source
    assert "subprocess" not in source
    assert "pull_request_target" not in source
    assert 'REVIEW_EVENT = "COMMENT"' in source


def test_parse_next_link(reviewer):
    header = (
        '<https://api.github.com/repos/o/r/pulls/1/files?page=2>; rel="next", '
        '<https://api.github.com/repos/o/r/pulls/1/files?page=3>; rel="last"'
    )
    assert reviewer.parse_next_link(header).endswith("page=2")
    assert reviewer.parse_next_link(None) is None


def test_model_identity_comes_from_secrets():
    text = WORKFLOW.read_text()
    assert "MODEL: ${{ secrets.MODEL }}" in text
    assert "MODEL_API_KEY: ${{ secrets.MODEL_API_KEY }}" in text
    assert "MODEL_API_URL: ${{ secrets.MODEL_API_URL }}" in text


def test_reviewer_sources_use_generic_names():
    vendor = "gr" + "ok"
    provider = "x" + "ai"
    for path in (SCRIPT, WORKFLOW):
        lowered = path.read_text().lower()
        assert vendor not in lowered
        assert provider not in lowered
        assert provider.replace("ai", ".ai") not in lowered

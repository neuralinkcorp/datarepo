import http.client
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "code_review.py"


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
    def __init__(self, routes, threads=None):
        self.repo = "neuralinkcorp/datarepo"
        self.routes = routes
        self.posts = []
        self.threads = list(threads or [])
        self.resolved = []

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

    def list_unresolved_bot_threads(self, number, bot_login):
        return [
            dict(thread)
            for thread in self.threads
            if thread.get("id") not in self.resolved
        ]

    def resolve_review_thread(self, thread_id):
        self.resolved.append(thread_id)


PR_ROUTES = {
    "/repos/neuralinkcorp/datarepo/pulls/12": {
        "number": 12,
        "title": "Fix nulls",
        "body": "Handle IS NULL",
        "draft": False,
        "user": {"login": "zack-dev-cm"},
        "head": {"sha": "abc"},
    },
    "/repos/neuralinkcorp/datarepo/pulls/12/reviews": [],
    "/repos/neuralinkcorp/datarepo/pulls/12/files": [
        {
            "filename": "src/datarepo/core/tables/clickhouse_table.py",
            "status": "modified",
            "patch": SAMPLE_PATCH,
        }
    ],
}

REVIEW_ENV = {
    "GITHUB_TOKEN": "t",
    "MODEL_API_KEY": "test-key",
    "MODEL": "test-model",
    "MODEL_API_URL": "https://example.invalid/v1/chat/completions",
    "REVIEW_PROMPT": "Review for correctness and tests.",
}

PR_EVENT = {
    "workflow_run": {
        "event": "pull_request",
        "head_sha": "abc",
        "pull_requests": [{"number": 12}],
    }
}


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


def test_select_inline_comments_excludes_previous_lines(reviewer):
    valid = {"a.py": {10, 11, 12}}
    raw = [
        {"path": "a.py", "line": 10, "severity": "high", "body": "old finding"},
        {"path": "a.py", "line": 12, "severity": "high", "body": "new finding"},
    ]
    selected = reviewer.select_inline_comments(raw, valid, exclude={("a.py", 10)})
    assert [c["line"] for c in selected] == [12]


def test_parse_model_output_strips_fences(reviewer):
    text = """```json
{"summary": "Looks fine.", "comments": [], "resolved": [1, 2]}
```"""
    parsed = reviewer.parse_model_output(text)
    assert parsed["summary"] == "Looks fine."
    assert parsed["comments"] == []
    assert parsed["resolved"] == [1, 2]


def test_parse_resolved_ids_filters_invalid(reviewer):
    assert reviewer.parse_resolved_ids([1, 1, 99, "2", "nope", 0], 3) == [1, 2]
    assert reviewer.parse_resolved_ids({"id": 1}, 3) == []
    assert reviewer.parse_resolved_ids([{"id": 3}], 3) == [3]


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


def test_resolve_bot_login_prefers_env_then_slug(reviewer):
    github = FakeGitHub({})
    assert (
        reviewer.resolve_bot_login({"REVIEW_BOT_LOGIN": "custom[bot]"}, github)
        == "custom[bot]"
    )
    assert (
        reviewer.resolve_bot_login({"APP_SLUG": "neuralink-code-review-bot"}, github)
        == "neuralink-code-review-bot[bot]"
    )
    assert reviewer.resolve_bot_login({}, github) == "code-review-bot[bot]"


def test_run_skips_without_secrets(reviewer):
    message = reviewer.run({}, {}, FakeGitHub({}), lambda *args: "")
    assert message == "skip: required secrets are not configured"


def test_run_posts_comment_review(reviewer):
    github = FakeGitHub(PR_ROUTES)

    def complete(api_key, model, system, user):
        assert api_key == "test-key"
        assert "clickhouse_table.py" in user
        assert REVIEW_ENV["REVIEW_PROMPT"] in system
        assert "resolved" in system
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

    message = reviewer.run(REVIEW_ENV, PR_EVENT, github, complete)
    assert "posted review 99" in message
    assert "1 inline comments, 0 resolved" in message
    assert len(github.posts) == 1
    payload = github.posts[0][1]
    assert payload["event"] == "COMMENT"
    assert payload["commit_id"] == "abc"
    assert payload["comments"][0]["line"] == 12
    assert github.resolved == []


def test_run_resolves_addressed_previous_comments(reviewer):
    github = FakeGitHub(
        PR_ROUTES,
        threads=[
            {
                "id": "TH_ADDRESSED",
                "path": "src/datarepo/core/tables/clickhouse_table.py",
                "line": 12,
                "body": "filters is rebound after query is built",
            },
            {
                "id": "TH_STILL_OPEN",
                "path": "src/datarepo/core/tables/clickhouse_table.py",
                "line": 13,
                "body": "needs a test for the null path",
            },
        ],
    )

    def complete(api_key, model, system, user):
        assert "[1] src/datarepo/core/tables/clickhouse_table.py:12" in user
        assert "[2] src/datarepo/core/tables/clickhouse_table.py:13" in user
        return json.dumps(
            {
                "summary": "The null-filter issue is fixed.",
                "comments": [
                    {
                        "path": "src/datarepo/core/tables/clickhouse_table.py",
                        "line": 12,
                        "severity": "high",
                        "body": "should not re-post the addressed finding",
                    },
                    {
                        "path": "src/datarepo/core/tables/clickhouse_table.py",
                        "line": 13,
                        "severity": "medium",
                        "body": "should not re-post the still-open finding",
                    },
                    {
                        "path": "src/datarepo/core/tables/clickhouse_table.py",
                        "line": 11,
                        "severity": "low",
                        "body": "new nit on a different line",
                    },
                ],
                "resolved": [1],
            }
        )

    message = reviewer.run(REVIEW_ENV, PR_EVENT, github, complete)
    assert "1 inline comments, 1 resolved" in message
    assert github.resolved == ["TH_ADDRESSED"]
    payload = github.posts[0][1]
    assert [c["line"] for c in payload["comments"]] == [11]
    assert "Resolved 1 previous comment(s) as addressed." in payload["body"]


def test_run_continues_when_previous_comments_fail_to_load(reviewer):
    github = FakeGitHub(PR_ROUTES)

    def boom(number, bot_login):
        raise RuntimeError("graphql down")

    github.list_unresolved_bot_threads = boom

    def complete(api_key, model, system, user):
        assert "Previous unresolved comments" not in user
        return json.dumps({"summary": "Looks fine.", "comments": []})

    message = reviewer.run(REVIEW_ENV, PR_EVENT, github, complete)
    assert "posted review 99" in message
    assert "0 resolved" in message


def test_list_unresolved_bot_threads_keeps_bot_open_threads(reviewer):
    response = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {
                                "id": "TH_KEEP",
                                "isResolved": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "author": {"login": "code-review-bot[bot]"},
                                            "body": "null filter dropped",
                                            "path": "a.py",
                                            "line": 12,
                                            "originalLine": 12,
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "TH_RESOLVED",
                                "isResolved": True,
                                "comments": {
                                    "nodes": [
                                        {
                                            "author": {"login": "code-review-bot[bot]"},
                                            "body": "old",
                                            "path": "a.py",
                                            "line": 1,
                                            "originalLine": 1,
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "TH_HUMAN",
                                "isResolved": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "author": {"login": "alice"},
                                            "body": "please fix",
                                            "path": "a.py",
                                            "line": 3,
                                            "originalLine": 3,
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "TH_FALLBACK_LINE",
                                "isResolved": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "author": {"login": "code-review-bot[bot]"},
                                            "body": "outdated line",
                                            "path": "b.py",
                                            "line": None,
                                            "originalLine": 40,
                                        }
                                    ]
                                },
                            },
                        ],
                    }
                }
            }
        }
    }

    def requester(url, method="GET", headers=None, payload=None, timeout=30):
        assert url.endswith("/graphql")
        return 200, response, {}

    github = reviewer.GitHubClient("t", "neuralinkcorp/datarepo", requester=requester)
    threads = github.list_unresolved_bot_threads(12, "code-review-bot[bot]")
    assert [item["id"] for item in threads] == ["TH_KEEP", "TH_FALLBACK_LINE"]
    assert threads[0]["line"] == 12
    assert threads[1]["line"] == 40


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


def test_parse_next_link(reviewer):
    header = (
        '<https://api.github.com/repos/o/r/pulls/1/files?page=2>; rel="next", '
        '<https://api.github.com/repos/o/r/pulls/1/files?page=3>; rel="last"'
    )
    assert reviewer.parse_next_link(header).endswith("page=2")
    assert reviewer.parse_next_link(None) is None


def model_response():
    return {"choices": [{"message": {"content": '{"summary": "ok"}'}}]}


class FakeResponse:
    def __init__(self, *, lines=(), raw=b"", headers=None, status=200):
        self.lines = lines
        self.raw = raw
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.raw


def test_parse_chat_stream_joins_content_and_ignores_non_content_chunks(reviewer):
    lines = [
        b": keep-alive\n",
        b"\n",
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n',
        b'data:{"choices": [{"delta": {"reasoning_content": "think"}}]}\n',
        b'data: {"choices": []}\n',
        b'data: {"choices": [{"delta": {"content": "hello"}}]}\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n',
        b"data: [DONE]\n",
    ]

    assert reviewer.parse_chat_stream(lines) == "hello world"


def test_parse_chat_stream_accepts_finish_reason_without_done(reviewer):
    lines = [
        b'data: {"choices": [{"delta": {"content": "ok"}, '
        b'"finish_reason": "stop"}]}\n'
    ]

    assert reviewer.parse_chat_stream(lines) == "ok"


def test_parse_chat_stream_rejects_incomplete_and_error_streams(reviewer):
    with pytest.raises(reviewer.TransportError, match="before completion"):
        reviewer.parse_chat_stream(
            [b'data: {"choices": [{"delta": {"content": "x"}}]}\n']
        )
    with pytest.raises(reviewer.TransportError, match="returned an error"):
        reviewer.parse_chat_stream([b'data: {"error": {"message": "bad"}}\n'])


def test_http_stream_chat_reads_sse_response(reviewer):
    response = FakeResponse(
        lines=[
            b'data: {"choices": [{"delta": {"content": "hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            b"data: [DONE]\n",
        ],
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )

    status, body, headers = reviewer.http_stream_chat(
        "https://example.invalid",
        opener=lambda *args, **kwargs: response,
    )

    assert status == 200
    assert body == {"choices": [{"message": {"content": "hello world"}}]}
    assert headers == {"Content-Type": "text/event-stream; charset=utf-8"}


def test_http_stream_chat_accepts_plain_json_response(reviewer):
    response = FakeResponse(
        raw=json.dumps(model_response()).encode(),
        headers={"Content-Type": "application/json"},
    )

    status, body, _ = reviewer.http_stream_chat(
        "https://example.invalid",
        opener=lambda *args, **kwargs: response,
    )

    assert status == 200
    assert body == model_response()


def test_http_stream_chat_returns_http_error_status(reviewer):
    error = urllib.error.HTTPError(
        "https://example.invalid", 503, "busy", {}, io.BytesIO(b'{"message": "busy"}')
    )

    status, body, headers = reviewer.http_stream_chat(
        "https://example.invalid",
        opener=lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    assert (status, body, headers) == (503, {"message": "busy"}, {})


def test_http_stream_chat_wraps_mid_stream_disconnect(reviewer):
    response = FakeResponse(
        lines=(
            line for line in [b'data: {"choices": [{"delta": {"content": "x"}}]}\n']
        ),
        headers={"Content-Type": "text/event-stream"},
    )

    def broken_lines():
        yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n'
        raise http.client.RemoteDisconnected("closed")

    response.lines = broken_lines()
    with pytest.raises(reviewer.TransportError, match="RemoteDisconnected"):
        reviewer.http_stream_chat(
            "https://example.invalid",
            opener=lambda *args, **kwargs: response,
        )


def test_http_stream_chat_enforces_deadline_while_reading(reviewer):
    response = FakeResponse(
        lines=[b'data: {"choices": [{"delta": {"content": "x"}}]}\n'],
        headers={"Content-Type": "text/event-stream"},
    )

    with pytest.raises(reviewer.ModelRequestError, match="budget exhausted"):
        reviewer.http_stream_chat(
            "https://example.invalid",
            opener=lambda *args, **kwargs: response,
            clock=iter([1]).__next__,
            deadline=1,
        )


def test_complete_chat_retries_incomplete_stream_then_succeeds(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    payloads = []

    def incomplete():
        return reviewer.parse_chat_stream(
            [b'data: {"choices": [{"delta": {"content": "x"}}]}\n']
        )

    responses = [incomplete, lambda: (200, model_response(), {})]

    def requester(*args, **kwargs):
        payloads.append(dict(kwargs["payload"]))
        return responses.pop(0)()

    assert (
        reviewer.complete_chat(
            "key",
            "model",
            "system",
            "user",
            requester=requester,
            sleep=lambda _: None,
            jitter=lambda *_: 0,
        )
        == '{"summary": "ok"}'
    )
    assert len(payloads) == 2
    assert all(payload["stream"] is True for payload in payloads)


def test_complete_chat_retries_remote_disconnect_then_succeeds(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    responses = [http.client.RemoteDisconnected("closed"), (200, model_response(), {})]
    sleeps = []

    def requester(*args, **kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    assert (
        reviewer.complete_chat(
            "key", "model", "system", "user", requester=requester, sleep=sleeps.append
        )
        == '{"summary": "ok"}'
    )
    assert len(sleeps) == 1


def test_complete_chat_retries_503_then_succeeds(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    responses = [(503, {"message": "busy"}, {}), (200, model_response(), {})]

    assert (
        reviewer.complete_chat(
            "key",
            "model",
            "system",
            "user",
            requester=lambda *args, **kwargs: responses.pop(0),
            sleep=lambda _: None,
            jitter=lambda *_: 0,
        )
        == '{"summary": "ok"}'
    )


def test_complete_chat_honors_retry_after(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    responses = [(429, {}, {"Retry-After": "7"}), (200, model_response(), {})]
    sleeps = []

    reviewer.complete_chat(
        "key",
        "model",
        "system",
        "user",
        requester=lambda *args, **kwargs: responses.pop(0),
        sleep=sleeps.append,
    )
    assert sleeps == [7]


def test_complete_chat_does_not_retry_400(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    calls = []

    with pytest.raises(reviewer.ModelRequestError, match="HTTP 400"):
        reviewer.complete_chat(
            "key",
            "model",
            "system",
            "user",
            requester=lambda *args, **kwargs: calls.append(1) or (400, {}, {}),
            sleep=lambda _: None,
        )
    assert calls == [1]


def test_complete_chat_exhausted_error_hides_model_url(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])

    with pytest.raises(reviewer.ModelRequestError) as exc_info:
        reviewer.complete_chat(
            "key",
            "model",
            "system",
            "user",
            requester=lambda *args, **kwargs: (503, {}, {}),
            sleep=lambda _: None,
            jitter=lambda *_: 0,
        )
    assert "after 3 attempts" in str(exc_info.value)
    assert REVIEW_ENV["MODEL_API_URL"] not in str(exc_info.value)


def test_complete_chat_fails_when_budget_exhausted(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    clock = iter([0, 0, 2]).__next__

    with pytest.raises(reviewer.ModelRequestError, match="budget exhausted"):
        reviewer.complete_chat(
            "key",
            "model",
            "system",
            "user",
            requester=lambda *args, **kwargs: (503, {}, {}),
            sleep=lambda _: None,
            clock=clock,
            jitter=lambda *_: 0,
            budget=1,
        )


def test_github_get_retries_503(reviewer):
    responses = [(503, {}, {}), (200, {"ok": True}, {})]
    sleeps = []
    github = reviewer.GitHubClient(
        "t",
        "neuralinkcorp/datarepo",
        requester=lambda *args, **kwargs: responses.pop(0),
        sleep=sleeps.append,
        jitter=lambda *_: 0,
    )

    assert github.get_json("/repos/neuralinkcorp/datarepo/pulls/12") == {"ok": True}
    assert sleeps == [1]


def test_post_review_avoids_duplicate_after_transport_error(reviewer):
    path = "/repos/neuralinkcorp/datarepo/pulls/12/reviews"

    class GitHub:
        repo = "neuralinkcorp/datarepo"

        def __init__(self):
            self.posts = 0

        def post_json(self, unused_path, payload):
            self.posts += 1
            raise reviewer.AmbiguousPostError("transport failed")

        def get_all(self, unused_path):
            return [{"user": {"login": "code-review-bot[bot]"}, "commit_id": "abc"}]

    github = GitHub()
    posted = reviewer.post_review(github, 12, {"commit_id": "abc"})
    assert posted["id"] == "existing"
    assert github.posts == 1


def test_post_review_reposts_after_transport_error_when_missing(reviewer):
    class GitHub:
        repo = "neuralinkcorp/datarepo"

        def __init__(self):
            self.posts = 0

        def post_json(self, unused_path, payload):
            self.posts += 1
            if self.posts == 1:
                raise reviewer.AmbiguousPostError("transport failed")
            return 200, {"id": 42}

        def get_all(self, unused_path):
            return []

    github = GitHub()
    assert reviewer.post_review(github, 12, {"commit_id": "abc"}) == {"id": 42}
    assert github.posts == 2


def test_complete_chat_retries_response_format_fallback(reviewer, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", REVIEW_ENV["MODEL_API_URL"])
    payloads = []
    responses = [
        (400, {"message": "response_format unsupported"}, {}),
        (200, model_response(), {}),
    ]

    def requester(*args, **kwargs):
        payloads.append(dict(kwargs["payload"]))
        return responses.pop(0)

    assert (
        reviewer.complete_chat("key", "model", "system", "user", requester=requester)
        == '{"summary": "ok"}'
    )
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]

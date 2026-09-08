#!/usr/bin/env python3
"""Review a pull request and post a GitHub COMMENT review.

Runs from Actions on the default branch. Fetches the diff over the API and
never checks out or executes pull-request code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping

LOGGER = logging.getLogger("code_review")

GITHUB_API = "https://api.github.com"
REQUIRED_SECRETS = (
    "GITHUB_TOKEN",
    "MODEL_API_KEY",
    "MODEL",
    "MODEL_API_URL",
    "REVIEW_PROMPT",
)
MAX_DIFF_CHARS = 200_000
MAX_INLINE_COMMENTS = 8
MAX_COMMENT_CHARS = 8_000
MAX_PREVIOUS_COMMENTS = 30
REVIEW_EVENT = "COMMENT"
FOOTER = (
    "*Posted by the code review bot. This is an automated review, "
    "not a maintainer approval.*"
)
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
SKIP_PR_EVENTS = frozenset({"pull_request"})
OUTPUT_INSTRUCTIONS = """Return a JSON object with this shape:
{
  "summary": "markdown review summary (2-8 sentences)",
  "comments": [
    {
      "path": "relative/path.py",
      "line": 12,
      "severity": "high",
      "body": "specific finding and why it matters"
    }
  ],
  "resolved": [1, 2]
}
severity must be high, medium, or low.
comments: at most 8 inline comments on RIGHT-side diff lines; use exact paths from the diff.
resolved: 1-based ids of previous unresolved bot comments that the current diff fully addresses. Use [] if none.
Do not re-comment previous unresolved findings unless the new diff introduces a distinct issue.
Do not approve or request changes as a GitHub review event.
"""
THREADS_QUERY = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          comments(first: 1) {
            nodes {
              author { login }
              body
              path
              line
              originalLine
            }
          }
        }
      }
    }
  }
}
"""
RESOLVE_MUTATION = """
mutation ($id: ID!) {
  resolveReviewThread(input: {threadId: $id}) {
    thread { isResolved }
  }
}
"""


class GitHubError(RuntimeError):
    def __init__(self, status: int, path: str, body: Any):
        super().__init__(f"GitHub API {status} for {path}: {body}")
        self.status = status
        self.body = body


def http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: Any = None,
    timeout: int = 30,
) -> tuple[int, Any, dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            parsed = json.loads(raw.decode("utf-8")) if raw else None
            return resp.status, parsed, {k: v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except json.JSONDecodeError:
            parsed = {"message": raw.decode("utf-8", errors="replace")}
        return exc.code, parsed, {k: v for k, v in exc.headers.items()}


def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">")
        if start != -1 and end != -1:
            return part[start + 1 : end]
    return None


def normalize_login(login: str) -> str:
    return login.lower().removesuffix("[bot]")


class GitHubClient:
    def __init__(
        self, token: str, repo: str, requester: Callable[..., Any] | None = None
    ):
        self.token = token
        self.repo = repo
        self._requester = requester or http_json

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "code-review-bot",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: Any = None,
        timeout: int = 30,
    ) -> tuple[int, Any, dict[str, str]]:
        if path.startswith("https://"):
            url = path
        else:
            url = GITHUB_API + path
            if params:
                url += "?" + urllib.parse.urlencode(params, doseq=True)
        headers = self._headers()
        if payload is not None:
            headers["Content-Type"] = "application/json"
        return self._requester(
            url, method=method, headers=headers, payload=payload, timeout=timeout
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        status, body, _ = self.request("GET", path, params=params)
        if status >= 400:
            raise GitHubError(status, path, body)
        return body

    def get_all(self, path: str, params: dict[str, Any] | None = None) -> list[Any]:
        query = dict(params or {})
        query.setdefault("per_page", 100)
        items: list[Any] = []
        next_path: str | None = path
        next_params: dict[str, Any] | None = query
        while next_path:
            status, body, headers = self.request("GET", next_path, params=next_params)
            if status >= 400:
                raise GitHubError(status, next_path, body)
            if not isinstance(body, list):
                raise GitHubError(status, next_path, body)
            items.extend(body)
            next_url = parse_next_link(headers.get("Link") or headers.get("link"))
            next_path = next_url
            next_params = None
        return items

    def post_json(self, path: str, payload: Any) -> tuple[int, Any]:
        status, body, _ = self.request("POST", path, payload=payload, timeout=60)
        return status, body

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> Any:
        status, body, _ = self.request(
            "POST",
            "/graphql",
            payload={"query": query, "variables": variables or {}},
            timeout=60,
        )
        if status >= 400:
            raise GitHubError(status, "/graphql", body)
        errors = (body or {}).get("errors")
        if errors:
            raise GitHubError(status, "/graphql", errors)
        return (body or {}).get("data")

    def list_unresolved_bot_threads(
        self, number: int, bot_login: str
    ) -> list[dict[str, Any]]:
        owner, name = self.repo.split("/", 1)
        bot_norm = normalize_login(bot_login)
        threads: list[dict[str, Any]] = []
        cursor = None
        while True:
            data = self.graphql(
                THREADS_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "number": number,
                    "cursor": cursor,
                },
            )
            conn = (
                ((data or {}).get("repository") or {}).get("pullRequest") or {}
            ).get("reviewThreads") or {}
            for node in conn.get("nodes") or []:
                if not node or node.get("isResolved"):
                    continue
                comments = (node.get("comments") or {}).get("nodes") or []
                if not comments:
                    continue
                first = comments[0] or {}
                author = ((first.get("author") or {}).get("login") or "")
                if normalize_login(author) != bot_norm:
                    continue
                line = first.get("line")
                if line is None:
                    line = first.get("originalLine")
                threads.append(
                    {
                        "id": node.get("id"),
                        "path": first.get("path") or "",
                        "line": line,
                        "body": first.get("body") or "",
                    }
                )
                if len(threads) >= MAX_PREVIOUS_COMMENTS:
                    return threads
            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            cursor = page.get("endCursor")
            if not cursor:
                break
        return threads

    def resolve_review_thread(self, thread_id: str) -> None:
        self.graphql(RESOLVE_MUTATION, {"id": thread_id})


def right_side_lines(patch: str | None) -> set[int]:
    """Return RIGHT-side file line numbers that GitHub will accept comments on."""
    lines: set[int] = set()
    if not patch:
        return lines
    new_line = 0
    for raw in patch.splitlines():
        match = HUNK_RE.match(raw)
        if match:
            new_line = int(match.group(3))
            continue
        if raw.startswith("\\"):
            continue
        if raw.startswith("+"):
            lines.add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            continue
        else:
            lines.add(new_line)
            new_line += 1
    return lines


def valid_comment_lines(files: list[dict[str, Any]]) -> dict[str, set[int]]:
    valid: dict[str, set[int]] = {}
    for file_info in files:
        path = file_info.get("filename")
        if not path:
            continue
        lines = right_side_lines(file_info.get("patch"))
        if lines:
            valid[path] = lines
    return valid


def format_diff(
    files: list[dict[str, Any]], limit: int = MAX_DIFF_CHARS
) -> tuple[str, list[str]]:
    chunks: list[str] = []
    omitted: list[str] = []
    used = 0
    for file_info in files:
        path = file_info.get("filename") or "unknown"
        patch = file_info.get("patch")
        status = file_info.get("status") or "modified"
        if not patch:
            omitted.append(path)
            continue
        block = f"### {path} ({status})\n```diff\n{patch}\n```\n"
        if used + len(block) > limit:
            omitted.append(path)
            continue
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks), omitted


def normalize_comment(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    path = raw.get("path") or raw.get("file") or raw.get("filename")
    body = (raw.get("body") or raw.get("comment") or "").strip()
    line = raw.get("line")
    if path is None or not body or line is None:
        return None
    try:
        line_no = int(line)
    except (TypeError, ValueError):
        return None
    if line_no < 1:
        return None
    severity = str(raw.get("severity") or "medium").lower()
    if severity not in SEVERITY_RANK:
        severity = "medium"
    return {
        "path": str(path),
        "line": line_no,
        "body": body[:MAX_COMMENT_CHARS],
        "severity": severity,
    }


def select_inline_comments(
    raw_comments: list[Any],
    valid_lines: dict[str, set[int]],
    limit: int = MAX_INLINE_COMMENTS,
    exclude: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    skip = exclude or set()
    for raw in raw_comments:
        if not isinstance(raw, Mapping):
            continue
        item = normalize_comment(raw)
        if item is None:
            continue
        key = (item["path"], item["line"])
        if key in seen or key in skip:
            continue
        lines = valid_lines.get(item["path"])
        if not lines or item["line"] not in lines:
            continue
        seen.add(key)
        selected.append(item)
    selected.sort(key=lambda c: (SEVERITY_RANK[c["severity"]], c["path"], c["line"]))
    return selected[:limit]


def parse_model_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    data = json.loads(stripped)
    if isinstance(data, list):
        return {"summary": "", "comments": data}
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    comments = data.get("comments") or data.get("inline_comments") or []
    if not isinstance(comments, list):
        comments = []
    summary = data.get("summary") or data.get("body") or ""
    if not isinstance(summary, str):
        summary = str(summary)
    return {
        "summary": summary.strip(),
        "comments": comments,
        "resolved": data.get("resolved") or data.get("addressed") or [],
    }


def build_review_payload(
    commit_id: str,
    summary: str,
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    body = (summary or "Automated review completed.").strip()
    if FOOTER not in body:
        body = f"{body}\n\n{FOOTER}"
    payload: dict[str, Any] = {
        "commit_id": commit_id,
        "body": body[:MAX_COMMENT_CHARS],
        "event": REVIEW_EVENT,
    }
    if comments:
        payload["comments"] = [
            {
                "path": item["path"],
                "line": item["line"],
                "side": "RIGHT",
                "body": item["body"],
            }
            for item in comments
        ]
    return payload


def configured(env: Mapping[str, str]) -> bool:
    return all(env.get(name) for name in REQUIRED_SECRETS)


def parse_resolved_ids(raw: Any, count: int) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    if not isinstance(raw, list):
        return ids
    for item in raw:
        value: Any = item
        if isinstance(item, Mapping):
            value = item.get("id") or item.get("index")
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number < 1 or number > count or number in seen:
            continue
        seen.add(number)
        ids.append(number)
    return ids


def threads_for_resolved_ids(
    threads: list[dict[str, Any]], resolved_ids: list[int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for index in resolved_ids:
        if 1 <= index <= len(threads) and threads[index - 1].get("id"):
            selected.append(threads[index - 1])
    return selected


def previous_comment_keys(threads: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in threads:
        path = item.get("path")
        line = item.get("line")
        if not path or line is None:
            continue
        try:
            keys.add((str(path), int(line)))
        except (TypeError, ValueError):
            continue
    return keys


def build_system_prompt(env: Mapping[str, str]) -> str:
    return env["REVIEW_PROMPT"].strip() + "\n\n" + OUTPUT_INSTRUCTIONS


def load_event(env: Mapping[str, str]) -> dict[str, Any]:
    path = env.get("GITHUB_EVENT_PATH")
    if not path:
        raise RuntimeError("GITHUB_EVENT_PATH is not set")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def workflow_run(event: Mapping[str, Any]) -> dict[str, Any]:
    run = event.get("workflow_run")
    if not isinstance(run, dict):
        return {}
    return run


def resolve_pull_number(github: GitHubClient, event: Mapping[str, Any]) -> int | None:
    run = workflow_run(event)
    for pr in run.get("pull_requests") or []:
        number = pr.get("number")
        if number:
            return int(number)

    head_sha = run.get("head_sha")
    if head_sha:
        pulls = github.get_json(f"/repos/{github.repo}/commits/{head_sha}/pulls")
        open_pulls = [pr for pr in pulls if pr.get("state") == "open"]
        if open_pulls:
            return int(open_pulls[0]["number"])

    head_repo = run.get("head_repository") or {}
    head_owner = (head_repo.get("owner") or {}).get("login") or ""
    if not head_owner and isinstance(head_repo.get("full_name"), str):
        head_owner = head_repo["full_name"].split("/", 1)[0]
    head_branch = run.get("head_branch")
    if head_owner and head_branch:
        pulls = github.get_json(
            f"/repos/{github.repo}/pulls",
            params={"head": f"{head_owner}:{head_branch}", "state": "open"},
        )
        if pulls:
            return int(pulls[0]["number"])
    return None


def skip_reason(
    *,
    event: Mapping[str, Any],
    pr: Mapping[str, Any],
    bot_login: str,
    reviews: list[Mapping[str, Any]],
    head_sha: str,
) -> str | None:
    run = workflow_run(event)
    if run.get("event") not in SKIP_PR_EVENTS:
        return f"triggering event is {run.get('event')!r}, not pull_request"
    if pr.get("draft"):
        return "pull request is a draft"
    author = (pr.get("user") or {}).get("login") or ""
    if author and normalize_login(author) == normalize_login(bot_login):
        return "pull request was opened by the review bot"
    bot_norm = normalize_login(bot_login)
    for review in reviews:
        user = (review.get("user") or {}).get("login") or ""
        if normalize_login(user) != bot_norm:
            continue
        if review.get("commit_id") == head_sha:
            return f"already reviewed {head_sha[:12]}"
    return None


def build_user_prompt(
    pr: Mapping[str, Any],
    diff_text: str,
    omitted: list[str],
    previous: list[dict[str, Any]] | None = None,
) -> str:
    title = pr.get("title") or ""
    body = (pr.get("body") or "").strip() or "(no description)"
    author = (pr.get("user") or {}).get("login") or "unknown"
    omitted_note = ""
    if omitted:
        omitted_note = "\nFiles with no (or truncated) patches:\n" + "\n".join(
            f"- {path}" for path in omitted
        )
    previous_note = ""
    if previous:
        lines = [
            "Previous unresolved comments from the code review bot.",
            "If a finding is fully addressed in the current diff, include its id in resolved.",
            "Do not repeat unresolved findings as new comments.",
        ]
        for index, item in enumerate(previous, start=1):
            path = item.get("path") or "unknown"
            line = item.get("line") if item.get("line") is not None else "?"
            text = (item.get("body") or "").strip().replace("\n", " ")
            lines.append(f"[{index}] {path}:{line} — {text}")
        previous_note = "\n" + "\n".join(lines) + "\n"
    return (
        f"Pull request #{pr.get('number')}: {title}\n"
        f"Author: {author}\n"
        f"Description:\n{body}\n\n"
        f"Diff:\n{diff_text}{omitted_note}{previous_note}\n"
    )


def complete_chat(
    api_key: str, model: str, system: str, user: str, timeout: int = 120
) -> str:
    api_url = os.environ.get("MODEL_API_URL")
    if not api_url:
        raise RuntimeError("MODEL_API_URL is not configured")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "code-review-bot",
    }
    status, body, _ = http_json(
        api_url,
        method="POST",
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    if status >= 400:
        message = str(body)
        if "response_format" in message.lower():
            payload.pop("response_format", None)
            status, body, _ = http_json(
                api_url,
                method="POST",
                headers=headers,
                payload=payload,
                timeout=timeout,
            )
        if status >= 400:
            raise RuntimeError(f"model API error {status}: {body}")
    choices = (body or {}).get("choices") or []
    if not choices:
        raise RuntimeError(f"model API returned no choices: {body}")
    content = choices[0].get("message", {}).get("content") or ""
    if not str(content).strip():
        raise RuntimeError("model API returned empty content")
    return str(content)


def post_review(github: GitHubClient, number: int, payload: dict[str, Any]) -> Any:
    path = f"/repos/{github.repo}/pulls/{number}/reviews"
    status, body = github.post_json(path, payload)
    if status == 422 and payload.get("comments"):
        LOGGER.warning("inline comments rejected (%s); retrying summary only", body)
        retry = dict(payload)
        retry.pop("comments", None)
        status, body = github.post_json(path, retry)
    if status >= 400:
        raise GitHubError(status, path, body)
    return body


def run(
    env: Mapping[str, str],
    event: Mapping[str, Any],
    github: GitHubClient,
    complete: Callable[[str, str, str, str], str],
) -> str:
    if not configured(env):
        return "skip: required secrets are not configured"

    number = resolve_pull_number(github, event)
    if number is None:
        return "skip: no pull request associated with this workflow run"

    pr = github.get_json(f"/repos/{github.repo}/pulls/{number}")
    me = github.get_json("/user")
    bot_login = me.get("login") or "code-review-bot[bot]"
    head_sha = (
        (pr.get("head") or {}).get("sha") or workflow_run(event).get("head_sha") or ""
    )
    reviews = github.get_all(f"/repos/{github.repo}/pulls/{number}/reviews")
    reason = skip_reason(
        event=event,
        pr=pr,
        bot_login=bot_login,
        reviews=reviews,
        head_sha=head_sha,
    )
    if reason:
        return f"skip: {reason}"

    files = github.get_all(f"/repos/{github.repo}/pulls/{number}/files")
    valid_lines = valid_comment_lines(files)
    diff_text, omitted = format_diff(files)
    if not diff_text.strip():
        return "skip: pull request has no reviewable diff"

    previous: list[dict[str, Any]] = []
    try:
        previous = github.list_unresolved_bot_threads(number, bot_login)
    except Exception as exc:
        LOGGER.warning("could not load previous comments: %s", exc)

    raw = complete(
        env["MODEL_API_KEY"],
        env["MODEL"],
        build_system_prompt(env),
        build_user_prompt(pr, diff_text, omitted, previous),
    )
    parsed = parse_model_output(raw)
    resolved_ids = parse_resolved_ids(parsed.get("resolved"), len(previous))
    to_resolve = threads_for_resolved_ids(previous, resolved_ids)
    comments = select_inline_comments(
        parsed["comments"],
        valid_lines,
        exclude=previous_comment_keys(previous),
    )
    summary = parsed["summary"]
    if to_resolve:
        summary = (
            f"{summary}\n\nResolved {len(to_resolve)} previous comment(s) "
            "as addressed."
        ).strip()
    payload = build_review_payload(head_sha, summary, comments)
    if payload["event"] != REVIEW_EVENT:
        raise RuntimeError("refusing to post a review that is not COMMENT")
    posted = post_review(github, number, payload)
    resolved_ok = 0
    for thread in to_resolve:
        try:
            github.resolve_review_thread(thread["id"])
            resolved_ok += 1
        except Exception as exc:
            LOGGER.warning("could not resolve thread %s: %s", thread.get("id"), exc)
    return (
        f"posted review {posted.get('id')} on PR #{number} "
        f"({len(comments)} inline comments, {resolved_ok} resolved)"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    env = os.environ
    try:
        event = load_event(env)
        repo = env.get("GITHUB_REPOSITORY")
        if not repo:
            raise RuntimeError("GITHUB_REPOSITORY is not set")
        github = GitHubClient(env.get("GITHUB_TOKEN") or "", repo)
        message = run(env, event, github, complete_chat)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("%s", message)
    return 0


if __name__ == "__main__":
    sys.exit(main())

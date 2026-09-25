from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import socket

from botocore.credentials import (
    Credentials,
    DeferredRefreshableCredentials,
    RefreshableCredentials,
)
from botocore.exceptions import ProfileNotFound
import pytest

from datarepo.core.tables import util


CASES = [
    pytest.param(
        util.get_storage_options,
        ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"),
        "aws_endpoint_url",
        "aws_region",
        id="storage-options",
    ),
    pytest.param(
        util.get_pyarrow_filesystem_args,
        ("access_key", "secret_key", "session_token"),
        "endpoint_override",
        "region",
        id="pyarrow-filesystem",
    ),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Unexpected network access in credential test")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(util.datarepo_config, "DEFAULT_AWS_PROFILE", None)


class Clock:
    def __init__(self, cross_after_first=False):
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.now = self.start
        self.calls = 0
        self.cross_after_first = cross_after_first

    def __call__(self):
        self.calls += 1
        if self.cross_after_first and self.calls > 1:
            self.now = max(self.now, self.start + timedelta(seconds=601))
        return self.now


def refreshable_session(*, cross_after_first=False, deferred=False):
    clock = Clock(cross_after_first)
    refreshes = []

    def refresh():
        refreshes.append(len(refreshes) + 1)
        generation = refreshes[-1]
        return {
            "access_key": f"test-key-{generation}",
            "secret_key": f"test-secret-{generation}",
            "token": f"test-token-{generation}",
            "expiry_time": (clock.now + timedelta(hours=1)).isoformat(),
        }

    if deferred:
        credentials = DeferredRefreshableCredentials(
            refresh_using=refresh, method="test", time_fetcher=clock
        )
    else:
        credentials = RefreshableCredentials(
            access_key="test-key-0",
            secret_key="test-secret-0",
            token="test-token-0",
            expiry_time=clock.now + timedelta(seconds=1200),
            refresh_using=refresh,
            method="test",
            time_fetcher=clock,
        )

    session = SimpleNamespace(
        region_name="us-east-1", get_credentials=lambda: credentials
    )
    return session, clock, refreshes


def generations(options, keys):
    return [int(options[key].rsplit("-", 1)[-1]) for key in keys]


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
def test_credentials_remain_coherent_across_refresh(helper, keys, endpoint, region):
    # The first clock read is outside botocore's refresh window; later reads
    # cross it. Replacement credentials stay valid for an hour.
    session, _, refreshes = refreshable_session(cross_after_first=True)
    options = helper(boto3_session=session, endpoint_url="https://storage.invalid")

    assert generations(options, keys) == [0, 0, 0]
    assert refreshes == []
    assert options[endpoint] == "https://storage.invalid"
    assert options[region] == "us-east-1"


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
def test_later_construction_refreshes_without_changing_prior_result(
    helper, keys, endpoint, region
):
    session, clock, refreshes = refreshable_session()
    first = helper(boto3_session=session)
    clock.now += timedelta(seconds=1201)
    second = helper(boto3_session=session)

    assert generations(first, keys) == [0, 0, 0]
    assert generations(second, keys) == [1, 1, 1]
    assert refreshes == [1]


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
def test_deferred_credentials(helper, keys, endpoint, region):
    session, _, refreshes = refreshable_session(deferred=True)

    options = helper(boto3_session=session)

    assert generations(options, keys) == [1, 1, 1]
    assert refreshes == [1]


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
@pytest.mark.parametrize("token", [None, "", "test-token-0"])
@pytest.mark.parametrize("region_name", [None, "us-east-1"])
def test_static_credentials(helper, keys, endpoint, region, token, region_name):
    credentials = Credentials("test-key-0", "test-secret-0", token)
    session = SimpleNamespace(
        region_name=region_name, get_credentials=lambda: credentials
    )

    options = helper(boto3_session=session, endpoint_url="https://storage.invalid")

    assert options[keys[0]] == "test-key-0"
    assert options[keys[1]] == "test-secret-0"
    assert options[endpoint] == "https://storage.invalid"
    if region_name is None:
        assert region not in options
    else:
        assert options[region] == region_name
    if helper is util.get_storage_options and not token:
        assert keys[2] not in options
    else:
        assert options[keys[2]] == (token or "")


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
def test_missing_credentials(helper, keys, endpoint, region, caplog):
    session = SimpleNamespace(region_name="us-east-1", get_credentials=lambda: None)

    options = helper(boto3_session=session, endpoint_url="https://storage.invalid")

    assert options == {endpoint: "https://storage.invalid"}
    assert "no credentials found" in caplog.text


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
def test_mandatory_refresh_failure_propagates(helper, keys, endpoint, region):
    clock = Clock()

    def broken_refresh():
        raise RuntimeError("credential provider failed")

    credentials = RefreshableCredentials(
        access_key="test-key-0",
        secret_key="test-secret-0",
        token="test-token-0",
        expiry_time=clock.now + timedelta(seconds=1),
        refresh_using=broken_refresh,
        method="test",
        time_fetcher=clock,
    )
    session = SimpleNamespace(
        region_name="us-east-1", get_credentials=lambda: credentials
    )

    with pytest.raises(RuntimeError, match="credential provider failed"):
        helper(boto3_session=session)


@pytest.mark.parametrize("helper,keys,endpoint,region", CASES)
def test_successful_construction_does_not_log_credentials(
    helper, keys, endpoint, region, caplog
):
    session, _, _ = refreshable_session(deferred=True)
    caplog.set_level("DEBUG")

    options = helper(boto3_session=session)

    for key in keys:
        assert options[key] not in caplog.text


def test_storage_options_uses_default_session(monkeypatch):
    credentials = Credentials("test-key-0", "test-secret-0", "test-token-0")
    session = SimpleNamespace(
        region_name="us-east-1", get_credentials=lambda: credentials
    )
    calls = []

    def create_session(**kwargs):
        calls.append(kwargs)
        return session

    monkeypatch.setattr(util.boto3, "Session", create_session)

    options = util.get_storage_options()

    assert calls == [{}]
    assert options["aws_access_key_id"] == "test-key-0"
    assert options["aws_region"] == "us-east-1"


def test_storage_options_uses_fallback_session_region(monkeypatch):
    monkeypatch.setattr(util.datarepo_config, "DEFAULT_AWS_PROFILE", "test-profile")
    empty = SimpleNamespace(region_name="us-east-1", get_credentials=lambda: None)
    fallback, _, _ = refreshable_session()
    fallback.region_name = "us-west-2"
    calls = []

    def create_session(**kwargs):
        calls.append(kwargs)
        return fallback

    monkeypatch.setattr(util.boto3, "Session", create_session)

    options = util.get_storage_options(boto3_session=empty)

    assert calls == [{"profile_name": "test-profile"}]
    assert options["aws_region"] == "us-west-2"
    assert options["aws_access_key_id"] == "test-key-0"


def test_storage_options_ignores_missing_profile(monkeypatch):
    monkeypatch.setattr(util.datarepo_config, "DEFAULT_AWS_PROFILE", "test-profile")

    def missing_profile(**kwargs):
        raise ProfileNotFound(profile="test-profile")

    monkeypatch.setattr(util.boto3, "Session", missing_profile)
    empty = SimpleNamespace(region_name=None, get_credentials=lambda: None)

    assert util.get_storage_options(boto3_session=empty) == {}


def test_pyarrow_does_not_create_default_or_fallback_session(monkeypatch):
    monkeypatch.setattr(util.datarepo_config, "DEFAULT_AWS_PROFILE", "test-profile")

    def deny(**kwargs):
        raise AssertionError("PyArrow helper must not create a session")

    monkeypatch.setattr(util.boto3, "Session", deny)
    expected = {"endpoint_override": "https://storage.invalid"}
    empty = SimpleNamespace(region_name=None, get_credentials=lambda: None)

    assert (
        util.get_pyarrow_filesystem_args(endpoint_url="https://storage.invalid")
        == expected
    )
    assert (
        util.get_pyarrow_filesystem_args(
            boto3_session=empty, endpoint_url="https://storage.invalid"
        )
        == expected
    )

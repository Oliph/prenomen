"""Tests for endpoint configuration: the key never comes from a committed file."""

import pytest

from prenomen.client import build_client, resolve_api_key


def test_the_named_environment_variable_supplies_the_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-from-env")
    assert resolve_api_key({"api_key_env": "MY_KEY"}) == "sk-from-env"


def test_the_environment_wins_over_a_literal_in_the_file(monkeypatch):
    # A stale literal left in a config must never silently override the real key.
    monkeypatch.setenv("MY_KEY", "sk-from-env")
    cfg = {"api_key_env": "MY_KEY", "api_key": "stale-literal"}
    assert resolve_api_key(cfg) == "sk-from-env"


def test_an_unset_variable_is_reported_by_name(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        resolve_api_key({"api_key_env": "MY_KEY"})
    assert "MY_KEY" in str(excinfo.value)


def test_an_empty_variable_counts_as_unset(monkeypatch):
    # An exported-but-empty variable is the usual shape of a typo in a job script.
    monkeypatch.setenv("MY_KEY", "")
    with pytest.raises(ValueError, match="MY_KEY"):
        resolve_api_key({"api_key_env": "MY_KEY"})


def test_a_literal_key_is_accepted_for_local_endpoints():
    assert resolve_api_key({"api_key": "dummy"}) == "dummy"


def test_neither_setting_falls_back_to_a_placeholder():
    # Local servers ignore authentication but the OpenAI client demands something.
    assert resolve_api_key({}) == "dummy"


def test_a_trailing_slash_on_the_base_url_is_dropped():
    client = build_client(
        {"api": {"base_url": "http://localhost:8000/v1/", "api_key": "x"}}
    )
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"


@pytest.mark.parametrize("model_config", [
    {},
    {"api": None},
    {"api": {}},
    {"api": {"api_key": "x"}},
    {"api": "https://api.openai.com/v1"},
])
def test_a_model_config_without_an_endpoint_is_refused_with_an_example(model_config):
    with pytest.raises(ValueError) as excinfo:
        build_client(model_config)
    assert "base_url" in str(excinfo.value)

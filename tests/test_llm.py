"""Retry behaviour of ChatLLM against the real openai exception classes (no network)."""
import types

import openai
import pytest

try:
    import httpx
except ImportError:  # newer openai releases use their own fork, httpx2
    import httpx2 as httpx

from unide import llm as L

RPD = ("Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-3.5-turbo-0125 in organization org-x on "
       "requests per day (RPD): Limit 10000, Used 10000, Requested 1. Please try again in 8.64s.', "
       "'code': 'rate_limit_exceeded'}}")


def _err(cls, status, message):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=req), body=None)


class FakeCompletions:
    def __init__(self, failures):
        self.failures, self.calls = list(failures), 0

    def create(self, **kw):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok": 1}'))])


def _llm(monkeypatch, failures, **kw):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    m = L.ChatLLM(**kw)
    fake = FakeCompletions(failures)
    m.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=fake))
    return m, fake


def test_parsing_helpers():
    assert L.suggested_wait(RPD) == pytest.approx(8.64)
    assert L.suggested_wait("Please try again in 896ms.") == pytest.approx(0.896)
    assert L.suggested_wait("Please try again in 1m30s.") == pytest.approx(90)
    assert L.suggested_wait("no hint") is None
    assert L.limit_kind(RPD) == "requests-per-day" and L.limit_kind("on tokens per min (TPM): x") == "tokens-per-minute"


def test_waits_through_many_rate_limits(monkeypatch):
    m, fake = _llm(monkeypatch, [_err(openai.RateLimitError, 429, RPD) for _ in range(25)])
    assert m("sys", "user") == '{"ok": 1}' and fake.calls == 26


def test_gives_up_after_max_rate_waits(monkeypatch):
    m, fake = _llm(monkeypatch, [_err(openai.RateLimitError, 429, RPD) for _ in range(5)], max_rate_waits=3)
    with pytest.raises(openai.RateLimitError):
        m("sys", "user")


def test_insufficient_quota_stops_immediately(monkeypatch):
    msg = "Error code: 429 - {'error': {'code': 'insufficient_quota', 'message': 'You exceeded your current quota'}}"
    m, fake = _llm(monkeypatch, [_err(openai.RateLimitError, 429, msg)] * 3)
    with pytest.raises(RuntimeError, match="out of credit"):
        m("sys", "user")
    assert fake.calls == 1


def test_bad_request_is_not_retried(monkeypatch):
    m, fake = _llm(monkeypatch, [_err(openai.BadRequestError, 400, "bad request")] * 3)
    with pytest.raises(openai.BadRequestError):
        m("sys", "user")
    assert fake.calls == 1

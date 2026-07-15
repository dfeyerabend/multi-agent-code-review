"""
Tests for the deterministic safety-guard logic in app.py: the rate limiter, client-IP
extraction, and the character counter.

The Gradio wiring, streaming, and download behaviour are verified manually in the browser —
these tests cover only the pure Python units.
"""

import time
import pytest
from app import RateLimiter, get_client_ip, render_char_count, CODE_CHAR_LIMIT


# === HELPERS ===

class _Clock:
    """Controllable replacement for time.time() so window logic is deterministic."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    """Freezes time.time() at a value the test can advance by setting clock.now."""
    fake = _Clock()
    monkeypatch.setattr(time, "time", fake)
    return fake


class _FakeRequest:
    """Minimal stand-in for gr.Request carrying only the attributes get_client_ip reads."""

    def __init__(self, headers: dict | None = None, client_host: str | None = None):
        self.headers = headers or {}
        self.client = type("Client", (), {"host": client_host})() if client_host else None


# === RateLimiter: per-IP hourly cap ===

def test_ratelimiter_allows_under_cap(clock):
    """A fresh IP is allowed and stays allowed up to the cap."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=50)
    for _ in range(4):
        limiter.record("1.2.3.4")
    allowed, message = limiter.check("1.2.3.4")

    assert allowed is True
    assert message == ""


def test_ratelimiter_blocks_at_hourly_cap(clock):
    """The fifth run exhausts a 5/hour cap, and the message names the hourly limit."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=50)
    for _ in range(5):
        limiter.record("1.2.3.4")
    allowed, message = limiter.check("1.2.3.4")

    assert allowed is False
    assert "Hourly" in message


def test_ratelimiter_hourly_window_resets(clock):
    """Runs older than one hour are pruned, so the IP may run again."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=50)
    for _ in range(5):
        limiter.record("1.2.3.4")

    clock.now += 3601   # advance just past the one-hour window
    allowed, _ = limiter.check("1.2.3.4")

    assert allowed is True


def test_ratelimiter_per_ip_isolation(clock):
    """One IP hitting its cap does not affect another IP."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=50)
    for _ in range(5):
        limiter.record("1.1.1.1")

    assert limiter.check("1.1.1.1")[0] is False
    assert limiter.check("2.2.2.2")[0] is True


def test_ratelimiter_remaining_counts_down(clock):
    """remaining starts at the cap and decreases with each recorded run."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=50)

    assert limiter.remaining("1.2.3.4") == 5
    limiter.record("1.2.3.4")
    limiter.record("1.2.3.4")
    assert limiter.remaining("1.2.3.4") == 3


# === RateLimiter: global daily cap ===

def test_ratelimiter_global_cap_blocks_new_ip(clock):
    """Once the global daily total is reached, even an unused IP is blocked."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=3)
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        limiter.record(ip)

    allowed, message = limiter.check("9.9.9.9")   # a fresh IP, but the day is capped
    assert allowed is False
    assert "Daily" in message


def test_ratelimiter_global_window_resets(clock):
    """Runs older than a day are pruned from the global tally."""
    limiter = RateLimiter(per_ip_hourly=5, global_daily=3)
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        limiter.record(ip)

    clock.now += 86401   # advance just past the one-day window
    allowed, _ = limiter.check("9.9.9.9")

    assert allowed is True


# === get_client_ip ===

def test_get_client_ip_none_returns_unknown():
    """A missing request yields 'unknown'."""
    assert get_client_ip(None) == "unknown"


def test_get_client_ip_uses_first_forwarded_hop():
    """The original client is the first entry of X-Forwarded-For."""
    request = _FakeRequest(headers={"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"})
    assert get_client_ip(request) == "1.1.1.1"


def test_get_client_ip_strips_whitespace():
    """Surrounding whitespace on the forwarded hop is trimmed."""
    request = _FakeRequest(headers={"x-forwarded-for": "  1.1.1.1  , 2.2.2.2"})
    assert get_client_ip(request) == "1.1.1.1"


def test_get_client_ip_falls_back_to_direct_client():
    """Without the header, the direct client host is used."""
    request = _FakeRequest(client_host="9.9.9.9")
    assert get_client_ip(request) == "9.9.9.9"


def test_get_client_ip_no_header_no_client_returns_unknown():
    """No forwarded header and no client resolves to 'unknown'."""
    assert get_client_ip(_FakeRequest()) == "unknown"


def test_get_client_ip_malformed_returns_unknown():
    """A malformed request object is caught and reported as 'unknown', never raised."""
    assert get_client_ip(object()) == "unknown"


# === render_char_count ===

def test_render_char_count_under_limit():
    """Under the limit, the count shows without a warning marker."""
    result = render_char_count("abc")
    assert "3 /" in result
    assert "⚠️" not in result


def test_render_char_count_at_limit_is_not_over():
    """Exactly at the limit is still acceptable (the cap is exclusive)."""
    assert "⚠️" not in render_char_count("x" * CODE_CHAR_LIMIT)


def test_render_char_count_over_limit_warns():
    """One character over the limit switches to the warning variant."""
    result = render_char_count("x" * (CODE_CHAR_LIMIT + 1))
    assert "⚠️" in result
    assert "over the limit" in result


def test_render_char_count_non_string_counts_zero():
    """A non-string input is treated as zero characters, not an error."""
    assert "0 /" in render_char_count(None)

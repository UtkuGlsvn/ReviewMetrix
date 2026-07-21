"""Tests for IP-based rate limiting on the scraping routes."""
import pytest

from reviewMetrix.ratelimit import RateLimiter, client_ip, limiter


FORM = {
    'google_id': 'com.mock.app', 'apple_name': 'mockapp',
    'country': 'us', 'language': 'en', 'max_reviews': '50',
    'complaint_threshold': '2', 'top_words': '10',
}


# --------------------------------------------------------------------------
# Limiter mechanics
# --------------------------------------------------------------------------

def test_allows_up_to_the_limit():
    rl = RateLimiter(limit=3, window=3600)
    for _ in range(3):
        allowed, _, _ = rl.check('ip')
        assert allowed is True

    allowed, remaining, retry = rl.check('ip')
    assert allowed is False
    assert remaining == 0
    assert retry > 0


def test_remaining_counts_down():
    rl = RateLimiter(limit=3, window=3600)
    assert rl.check('ip')[1] == 2
    assert rl.check('ip')[1] == 1
    assert rl.check('ip')[1] == 0


def test_cost_is_charged_in_units():
    """A multi-scrape request costs more than one unit."""
    rl = RateLimiter(limit=6, window=3600)

    allowed, remaining, _ = rl.check('ip', cost=6)
    assert allowed is True and remaining == 0

    assert rl.check('ip', cost=1)[0] is False


def test_request_over_budget_is_rejected_whole():
    """A request costing more than the remaining budget is refused outright."""
    rl = RateLimiter(limit=5, window=3600)
    rl.check('ip', cost=3)

    assert rl.check('ip', cost=3)[0] is False, 'would exceed the limit'
    assert rl.usage('ip') == 3, 'a rejected request must not be recorded'
    assert rl.check('ip', cost=2)[0] is True, 'a request that still fits is allowed'


def test_rejected_requests_do_not_extend_the_window():
    """Retrying while blocked must not push the reset further away."""
    rl = RateLimiter(limit=1, window=3600)
    rl.check('ip')

    first_retry = rl.check('ip')[2]
    for _ in range(5):
        rl.check('ip')
    assert rl.check('ip')[2] <= first_retry


def test_ips_are_tracked_separately():
    rl = RateLimiter(limit=1, window=3600)
    assert rl.check('1.1.1.1')[0] is True
    assert rl.check('1.1.1.1')[0] is False
    assert rl.check('2.2.2.2')[0] is True, 'another IP has its own budget'


def test_window_expiry_frees_budget():
    """Entries older than the window stop counting."""
    rl = RateLimiter(limit=1, window=0)   # everything is immediately stale
    assert rl.check('ip')[0] is True
    assert rl.check('ip')[0] is True


def test_stale_ips_are_evicted(monkeypatch):
    """Tracking must not grow without bound across many distinct IPs."""
    import reviewMetrix.ratelimit as rlmod
    monkeypatch.setattr(rlmod, 'MAX_TRACKED_KEYS', 10)

    rl = rlmod.RateLimiter(limit=5, window=0)  # every entry is instantly stale
    for i in range(50):
        rl.check(f'ip-{i}')

    assert len(rl._hits) <= 11, 'stale keys should have been evicted'


def test_reset_clears_counters():
    rl = RateLimiter(limit=1, window=3600)
    rl.check('ip')
    rl.reset()
    assert rl.check('ip')[0] is True


# --------------------------------------------------------------------------
# Client IP resolution
# --------------------------------------------------------------------------

def test_forwarded_header_ignored_by_default(app, monkeypatch):
    """X-Forwarded-For is spoofable, so it is only trusted when opted in."""
    monkeypatch.delenv('TRUST_PROXY', raising=False)
    with app.test_request_context('/', headers={'X-Forwarded-For': '9.9.9.9'},
                                  environ_base={'REMOTE_ADDR': '10.0.0.1'}):
        from flask import request
        assert client_ip(request) == '10.0.0.1'


def test_forwarded_header_used_when_trusted(app, monkeypatch):
    monkeypatch.setenv('TRUST_PROXY', '1')
    with app.test_request_context('/', headers={'X-Forwarded-For': '9.9.9.9, 10.0.0.5'},
                                  environ_base={'REMOTE_ADDR': '10.0.0.1'}):
        from flask import request
        assert client_ip(request) == '9.9.9.9', 'the original client comes first'


# --------------------------------------------------------------------------
# Route integration
# --------------------------------------------------------------------------

def test_analyze_blocked_after_limit(client, patched_fetch, monkeypatch):
    monkeypatch.setattr(limiter, 'limit', 2)

    assert client.post('/analyze', data=FORM).status_code == 200
    assert client.post('/analyze', data=FORM).status_code == 200

    resp = client.post('/analyze', data=FORM)
    assert resp.status_code == 429
    assert 'Rate limit reached' in resp.data.decode()
    assert 'Retry-After' in resp.headers


def test_compare_costs_two_units(client, patched_fetch, monkeypatch):
    monkeypatch.setattr(limiter, 'limit', 3)

    resp = client.post('/compare', data={**FORM, 'google_id_b': 'com.mock.b',
                                         'apple_name_b': 'mockb'})
    assert resp.status_code == 200
    # Two units spent, one left: a second comparison costs two and must fail
    resp = client.post('/compare', data={**FORM, 'google_id_b': 'com.mock.b',
                                         'apple_name_b': 'mockb'})
    assert resp.status_code == 429


def test_country_comparison_costs_one_unit_per_country(client, patched_fetch, monkeypatch):
    monkeypatch.setattr(limiter, 'limit', 5)

    resp = client.post('/compare-countries', data={**FORM, 'countries': 'us, gb, de'})
    assert resp.status_code == 200

    # Three units spent of five; another three-country run does not fit
    resp = client.post('/compare-countries', data={**FORM, 'countries': 'fr, it, es'})
    assert resp.status_code == 429


def test_index_is_not_rate_limited(client, monkeypatch):
    """Only the scraping routes are limited; the form itself stays reachable."""
    monkeypatch.setattr(limiter, 'limit', 1)
    for _ in range(5):
        assert client.get('/').status_code == 200


def test_empty_country_list_is_not_charged(client, patched_fetch, monkeypatch):
    """A validation error costs no scrape budget."""
    monkeypatch.setattr(limiter, 'limit', 2)

    resp = client.post('/compare-countries', data={**FORM, 'countries': ' , '})
    assert resp.status_code == 200
    assert 'at least one country code' in resp.data.decode()

    # Budget untouched, so two full analyses still fit
    assert client.post('/analyze', data=FORM).status_code == 200
    assert client.post('/analyze', data=FORM).status_code == 200

"""Tests for the request hardening: security headers, the cross-site POST
guard, store-code validation and the outbound iTunes request."""
import pandas as pd
import pytest

from test_routes import form


# --------------------------------------------------------------------------
# security headers
# --------------------------------------------------------------------------

@pytest.mark.parametrize('path', ['/', '/prompts'])
def test_security_headers_present(client, path):
    resp = client.get(path)
    assert resp.headers['X-Content-Type-Options'] == 'nosniff'
    assert resp.headers['X-Frame-Options'] == 'DENY'
    assert 'Referrer-Policy' in resp.headers
    csp = resp.headers['Content-Security-Policy']
    assert "frame-ancestors 'none'" in csp
    assert "form-action 'self'" in csp
    assert 'https://cdn.jsdelivr.net' in csp


def test_chart_library_is_pinned_with_integrity(client, patched_fetch):
    body = client.post('/analyze', data=form()).data.decode()
    assert 'cdn.jsdelivr.net/npm/apexcharts@' in body
    assert 'integrity="sha384-' in body


# --------------------------------------------------------------------------
# cross-site POST guard
# --------------------------------------------------------------------------

def test_cross_site_origin_is_rejected(client, patched_fetch):
    resp = client.post('/analyze', data=form(),
                       headers={'Origin': 'https://evil.example'})
    assert resp.status_code == 403


def test_cross_site_fetch_metadata_is_rejected(client, patched_fetch):
    resp = client.post('/analyze', data=form(),
                       headers={'Sec-Fetch-Site': 'cross-site'})
    assert resp.status_code == 403


def test_rejected_post_does_not_use_rate_limit(client, patched_fetch):
    from reviewMetrix.ratelimit import limiter
    client.post('/analyze', data=form(), headers={'Origin': 'https://evil.example'})
    assert limiter.usage('127.0.0.1') == 0


def test_same_origin_post_is_allowed(client, patched_fetch):
    resp = client.post('/analyze', data=form(),
                       headers={'Origin': 'http://localhost',
                                'Sec-Fetch-Site': 'same-origin'})
    assert resp.status_code == 200


# --------------------------------------------------------------------------
# store code validation
# --------------------------------------------------------------------------

def _capture_fetch(monkeypatch, reviews_df, summary_stats):
    from reviewMetrix import analyzer
    calls = []

    def capture(google_id, apple_name, country, lang, max_reviews):
        calls.append((country, lang))
        return reviews_df.copy(), summary_stats

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', capture)
    return calls


@pytest.mark.parametrize('country,language,expected', [
    ('US', 'en', ('us', 'en')),
    ('gb', 'zh-TW', ('gb', 'zh-TW')),
    ('us&x=', 'en', ('us', 'en')),
    ('de', '../x', ('de', 'en')),
    ('', '', ('us', 'en')),
])
def test_country_and_language_are_validated(client, monkeypatch, reviews_df,
                                            summary_stats, country, language, expected):
    calls = _capture_fetch(monkeypatch, reviews_df, summary_stats)
    client.post('/analyze', data=form(country=country, language=language))
    assert calls == [expected]


def test_compare_countries_drops_malformed_codes(client, monkeypatch, reviews_df, summary_stats):
    calls = _capture_fetch(monkeypatch, reviews_df, summary_stats)
    client.post('/compare-countries',
                data=form(countries='us, <b>, gbr, d e, ' + 'x' * 500 + ', de'))
    assert [c for c, _ in calls] == ['us', 'de']


# --------------------------------------------------------------------------
# outbound iTunes request
# --------------------------------------------------------------------------

def test_itunes_request_has_timeout_and_encoded_params(monkeypatch):
    from reviewMetrix import analyzer
    seen = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {'resultCount': 1, 'results': [{
                'trackName': 'X', 'trackViewUrl': 'javascript:alert(1)'}]}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return FakeResp()

    class FakeAppStore:
        def __init__(self, **kwargs):
            self.reviews = []

        def review(self, how_many):
            pass

    def failing(*args, **kwargs):
        raise RuntimeError('no network in tests')

    monkeypatch.setattr(analyzer.requests, 'get', fake_get)
    monkeypatch.setattr(analyzer, 'AppStore', FakeAppStore)
    monkeypatch.setattr(analyzer, 'reviews', failing)

    df, stats = analyzer.fetch_reviews_store('com.x', 'a&b c', 'us', 'en', 10)

    assert seen['timeout'] == analyzer.ITUNES_TIMEOUT_SECONDS
    assert '?' not in seen['url'], 'query values must go through params='
    assert seen['params']['term'] == 'a&b c'
    # a non-https store link is never rendered as an href
    assert stats['ios']['store_url'] is None
    assert isinstance(df, pd.DataFrame)


def test_google_store_url_encodes_package_id(monkeypatch):
    from reviewMetrix import analyzer

    monkeypatch.setattr(analyzer, 'reviews', lambda *a, **k: ([], None))
    monkeypatch.setattr(analyzer, 'google_app_details', lambda *a, **k: {'title': 'X'})
    monkeypatch.setattr(analyzer.requests, 'get',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')))
    monkeypatch.setattr(analyzer, 'AppStore',
                        lambda **k: (_ for _ in ()).throw(RuntimeError('offline')))

    _, stats = analyzer.fetch_reviews_store('com.x&hl="><b>', '', 'us', 'en', 10)
    assert stats['google']['store_url'] == (
        'https://play.google.com/store/apps/details?id=com.x%26hl%3D%22%3E%3Cb%3E')

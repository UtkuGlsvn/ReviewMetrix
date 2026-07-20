"""Tests for the scraping result cache.

The cache lives at module level; the autouse clear_cache fixture in conftest
empties it around every test.
"""
import pandas as pd
import pytest

from reviewMetrix import analyzer


@pytest.fixture
def counting_fetch(monkeypatch, reviews_df, summary_stats):
    """Replace fetch_reviews_store with a call-counting fake."""
    calls = []

    def fake_fetch(google_id, apple_name, country, lang, max_reviews):
        calls.append((google_id, apple_name, country, lang, max_reviews))
        return reviews_df.copy(), dict(summary_stats)

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', fake_fetch)
    return calls


ARGS = ('com.mock', 'mock', 'us', 'en', 50)


# --------------------------------------------------------------------------
# Basic cache behaviour
# --------------------------------------------------------------------------

def test_second_call_hits_cache(counting_fetch):
    analyzer.fetch_reviews_cached(*ARGS)
    analyzer.fetch_reviews_cached(*ARGS)

    assert len(counting_fetch) == 1, "the second call should come from cache"


def test_cached_result_equals_fresh_result(counting_fetch):
    df1, stats1 = analyzer.fetch_reviews_cached(*ARGS)
    df2, stats2 = analyzer.fetch_reviews_cached(*ARGS)

    pd.testing.assert_frame_equal(df1, df2)
    assert stats1 == stats2


def test_different_keys_are_cached_separately(counting_fetch):
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.b', 'b', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.a', 'a', 'gb', 'en', 50)   # different country
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 100)  # different review count

    assert len(counting_fetch) == 4
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)   # repeat of the first
    assert len(counting_fetch) == 4, "the same key should not scrape again"


# --------------------------------------------------------------------------
# TTL and forced refresh
# --------------------------------------------------------------------------

def test_expired_entry_is_refetched(counting_fetch):
    analyzer.fetch_reviews_cached(*ARGS, ttl=3600)
    # ttl=0 treats every entry as expired
    analyzer.fetch_reviews_cached(*ARGS, ttl=0)

    assert len(counting_fetch) == 2


def test_ttl_zero_bypasses_but_still_stores(counting_fetch):
    """A forced refresh skips the cache but still stores the fresh result."""
    analyzer.fetch_reviews_cached(*ARGS, ttl=0)
    assert len(counting_fetch) == 1

    analyzer.fetch_reviews_cached(*ARGS)  # normal TTL should hit the fresh entry
    assert len(counting_fetch) == 1


def test_force_refresh_in_build_app_report(counting_fetch):
    analyzer.build_app_report('com.mock', 'mock', 'us', 'en', 50, 2, 10)
    analyzer.build_app_report('com.mock', 'mock', 'us', 'en', 50, 2, 10)
    assert len(counting_fetch) == 1, "the second report should use the cache"

    analyzer.build_app_report('com.mock', 'mock', 'us', 'en', 50, 2, 10,
                              force_refresh=True)
    assert len(counting_fetch) == 2, "force_refresh should bypass the cache"


# --------------------------------------------------------------------------
# Empty results are never cached
# --------------------------------------------------------------------------

def test_empty_result_is_not_cached(monkeypatch):
    """A transient empty scrape must not be pinned for an hour."""
    calls = []

    def empty_fetch(*args):
        calls.append(args)
        return pd.DataFrame(), {'google': None, 'ios': None}

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', empty_fetch)

    analyzer.fetch_reviews_cached(*ARGS)
    analyzer.fetch_reviews_cached(*ARGS)

    assert len(calls) == 2, "an empty result should not be cached"
    assert analyzer.review_cache_info()['entries'] == 0


# --------------------------------------------------------------------------
# Isolation: mutating returned data must not corrupt the cache
# --------------------------------------------------------------------------

def test_mutating_returned_dataframe_does_not_corrupt_cache(counting_fetch):
    df1, _ = analyzer.fetch_reviews_cached(*ARGS)
    original_len = len(df1)

    df1.drop(df1.index[0], inplace=True)          # the caller corrupts the data
    df1['review'] = 'overwritten'

    df2, _ = analyzer.fetch_reviews_cached(*ARGS)
    assert len(df2) == original_len
    assert 'overwritten' not in df2['review'].values


def test_mutating_returned_stats_does_not_corrupt_cache(counting_fetch):
    _, stats1 = analyzer.fetch_reviews_cached(*ARGS)
    stats1['google']['rating'] = 0.0
    stats1['ios'] = None

    _, stats2 = analyzer.fetch_reviews_cached(*ARGS)
    assert stats2['google']['rating'] != 0.0
    assert stats2['ios'] is not None


# --------------------------------------------------------------------------
# LRU eviction and cache management
# --------------------------------------------------------------------------

def test_cache_evicts_oldest_beyond_max_entries(counting_fetch, monkeypatch):
    monkeypatch.setattr(analyzer, 'CACHE_MAX_ENTRIES', 3)

    for i in range(4):
        analyzer.fetch_reviews_cached(f'com.app{i}', f'app{i}', 'us', 'en', 50)

    assert analyzer.review_cache_info()['entries'] == 3

    # app0 went in first, so it should have been evicted and re-scraped
    before = len(counting_fetch)
    analyzer.fetch_reviews_cached('com.app0', 'app0', 'us', 'en', 50)
    assert len(counting_fetch) == before + 1


def test_recently_used_entry_survives_eviction(counting_fetch, monkeypatch):
    """A recently used entry survives eviction even if it went in earlier."""
    monkeypatch.setattr(analyzer, 'CACHE_MAX_ENTRIES', 2)

    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.b', 'b', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)  # refresh a
    analyzer.fetch_reviews_cached('com.c', 'c', 'us', 'en', 50)  # b should be evicted

    before = len(counting_fetch)
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)
    assert len(counting_fetch) == before, "a should still be cached"

    analyzer.fetch_reviews_cached('com.b', 'b', 'us', 'en', 50)
    assert len(counting_fetch) == before + 1, "b should have been evicted"


def test_clear_review_cache_empties_it(counting_fetch):
    analyzer.fetch_reviews_cached(*ARGS)
    assert analyzer.review_cache_info()['entries'] == 1

    analyzer.clear_review_cache()
    assert analyzer.review_cache_info()['entries'] == 0

    analyzer.fetch_reviews_cached(*ARGS)
    assert len(counting_fetch) == 2, "a cleared cache should scrape again"


# --------------------------------------------------------------------------
# Cache at the route level
# --------------------------------------------------------------------------

def test_repeated_analyze_requests_scrape_once(client, counting_fetch):
    data = {
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    }
    assert client.post('/analyze', data=data).status_code == 200
    assert client.post('/analyze', data=data).status_code == 200

    assert len(counting_fetch) == 1


def test_force_refresh_checkbox_bypasses_cache(client, counting_fetch):
    data = {
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    }
    client.post('/analyze', data=data)
    client.post('/analyze', data={**data, 'force_refresh': 'on'})

    assert len(counting_fetch) == 2


def test_country_comparison_caches_per_country(client, counting_fetch):
    data = {
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
        'countries': 'us, gb, de',
    }
    client.post('/compare-countries', data=data)
    assert len(counting_fetch) == 3, "one scrape per country"

    client.post('/compare-countries', data=data)
    assert len(counting_fetch) == 3, "the second request should be fully cached"

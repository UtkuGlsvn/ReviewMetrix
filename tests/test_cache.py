"""Scraping sonuç cache'inin testleri.

Cache modül düzeyinde yaşar; conftest'teki autouse clear_cache fixture'ı
her testten önce/sonra onu boşaltır.
"""
import pandas as pd
import pytest

from reviewMetrix import analyzer


@pytest.fixture
def counting_fetch(monkeypatch, reviews_df, summary_stats):
    """fetch_reviews_store'u sayaçlı sahte bir fonksiyonla değiştirir."""
    calls = []

    def fake_fetch(google_id, apple_name, country, lang, max_reviews):
        calls.append((google_id, apple_name, country, lang, max_reviews))
        return reviews_df.copy(), dict(summary_stats)

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', fake_fetch)
    return calls


ARGS = ('com.mock', 'mock', 'us', 'en', 50)


# --------------------------------------------------------------------------
# Temel cache davranışı
# --------------------------------------------------------------------------

def test_second_call_hits_cache(counting_fetch):
    analyzer.fetch_reviews_cached(*ARGS)
    analyzer.fetch_reviews_cached(*ARGS)

    assert len(counting_fetch) == 1, "ikinci çağrı cache'ten dönmeli"


def test_cached_result_equals_fresh_result(counting_fetch):
    df1, stats1 = analyzer.fetch_reviews_cached(*ARGS)
    df2, stats2 = analyzer.fetch_reviews_cached(*ARGS)

    pd.testing.assert_frame_equal(df1, df2)
    assert stats1 == stats2


def test_different_keys_are_cached_separately(counting_fetch):
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.b', 'b', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.a', 'a', 'gb', 'en', 50)   # farklı ülke
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 100)  # farklı adet

    assert len(counting_fetch) == 4
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)   # ilkinin tekrarı
    assert len(counting_fetch) == 4, "aynı anahtar yeniden scrape etmemeli"


# --------------------------------------------------------------------------
# TTL / zorunlu yenileme
# --------------------------------------------------------------------------

def test_expired_entry_is_refetched(counting_fetch):
    analyzer.fetch_reviews_cached(*ARGS, ttl=3600)
    # ttl=0 -> her kayıt süresi dolmuş sayılır
    analyzer.fetch_reviews_cached(*ARGS, ttl=0)

    assert len(counting_fetch) == 2


def test_ttl_zero_bypasses_but_still_stores(counting_fetch):
    """Zorunlu yenileme cache'i atlar ama taze sonucu yine de saklar."""
    analyzer.fetch_reviews_cached(*ARGS, ttl=0)
    assert len(counting_fetch) == 1

    analyzer.fetch_reviews_cached(*ARGS)  # normal TTL -> taze kayıttan dönmeli
    assert len(counting_fetch) == 1


def test_force_refresh_in_build_app_report(counting_fetch):
    analyzer.build_app_report('com.mock', 'mock', 'us', 'en', 50, 2, 10)
    analyzer.build_app_report('com.mock', 'mock', 'us', 'en', 50, 2, 10)
    assert len(counting_fetch) == 1, "ikinci rapor cache kullanmalı"

    analyzer.build_app_report('com.mock', 'mock', 'us', 'en', 50, 2, 10,
                              force_refresh=True)
    assert len(counting_fetch) == 2, "force_refresh cache'i atlamalı"


# --------------------------------------------------------------------------
# Boş sonuçlar cache'lenmemeli
# --------------------------------------------------------------------------

def test_empty_result_is_not_cached(monkeypatch):
    """Scraper geçici olarak boş dönerse bu bir saat boyunca sabitlenmemeli."""
    calls = []

    def empty_fetch(*args):
        calls.append(args)
        return pd.DataFrame(), {'google': None, 'ios': None}

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', empty_fetch)

    analyzer.fetch_reviews_cached(*ARGS)
    analyzer.fetch_reviews_cached(*ARGS)

    assert len(calls) == 2, "boş sonuç cache'lenmemeli"
    assert analyzer.review_cache_info()['entries'] == 0


# --------------------------------------------------------------------------
# İzolasyon: cache'ten dönen veri değiştirilse bile cache bozulmamalı
# --------------------------------------------------------------------------

def test_mutating_returned_dataframe_does_not_corrupt_cache(counting_fetch):
    df1, _ = analyzer.fetch_reviews_cached(*ARGS)
    original_len = len(df1)

    df1.drop(df1.index[0], inplace=True)          # çağıran taraf veriyi bozuyor
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
# LRU tahliye ve cache yönetimi
# --------------------------------------------------------------------------

def test_cache_evicts_oldest_beyond_max_entries(counting_fetch, monkeypatch):
    monkeypatch.setattr(analyzer, 'CACHE_MAX_ENTRIES', 3)

    for i in range(4):
        analyzer.fetch_reviews_cached(f'com.app{i}', f'app{i}', 'us', 'en', 50)

    assert analyzer.review_cache_info()['entries'] == 3

    # İlk giren (app0) tahliye edilmiş olmalı -> yeniden scrape edilir
    before = len(counting_fetch)
    analyzer.fetch_reviews_cached('com.app0', 'app0', 'us', 'en', 50)
    assert len(counting_fetch) == before + 1


def test_recently_used_entry_survives_eviction(counting_fetch, monkeypatch):
    """LRU: yakın zamanda kullanılan kayıt, eski olsa da tahliye edilmemeli."""
    monkeypatch.setattr(analyzer, 'CACHE_MAX_ENTRIES', 2)

    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.b', 'b', 'us', 'en', 50)
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)  # a'yı tazele
    analyzer.fetch_reviews_cached('com.c', 'c', 'us', 'en', 50)  # b tahliye olmalı

    before = len(counting_fetch)
    analyzer.fetch_reviews_cached('com.a', 'a', 'us', 'en', 50)
    assert len(counting_fetch) == before, "a hâlâ cache'te olmalı"

    analyzer.fetch_reviews_cached('com.b', 'b', 'us', 'en', 50)
    assert len(counting_fetch) == before + 1, "b tahliye edilmiş olmalı"


def test_clear_review_cache_empties_it(counting_fetch):
    analyzer.fetch_reviews_cached(*ARGS)
    assert analyzer.review_cache_info()['entries'] == 1

    analyzer.clear_review_cache()
    assert analyzer.review_cache_info()['entries'] == 0

    analyzer.fetch_reviews_cached(*ARGS)
    assert len(counting_fetch) == 2, "temizlikten sonra yeniden scrape edilmeli"


# --------------------------------------------------------------------------
# Route seviyesinde cache
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
    assert len(counting_fetch) == 3, "her ülke için bir kez scrape"

    client.post('/compare-countries', data=data)
    assert len(counting_fetch) == 3, "ikinci istek tamamen cache'ten gelmeli"

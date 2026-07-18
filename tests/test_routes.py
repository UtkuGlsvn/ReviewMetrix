"""Flask route'larının testleri.

Scraping katmanı (fetch_reviews_store) patched_fetch fixture'ı ile
değiştirildiği için testler ağ erişimi olmadan çalışır.
"""
import pandas as pd
import pytest

from reviewMetrix.routes import _avg_store_rating


BASE_FORM = {
    'google_id': 'com.mock.app',
    'apple_name': 'mockapp',
    'country': 'us',
    'language': 'en',
    'max_reviews': '50',
    'complaint_threshold': '2',
    'top_words': '10',
    'extra_stopwords': 'app,one',
}


def form(**overrides):
    data = dict(BASE_FORM)
    data.update(overrides)
    return data


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

def test_index_renders_form(client):
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'ReviewMetrix' in body
    assert 'Quick Preset' in body
    assert 'Compare Two Apps' in body
    assert 'Compare Countries' in body


# --------------------------------------------------------------------------
# /analyze
# --------------------------------------------------------------------------

def test_analyze_renders_full_dashboard(client, patched_fetch):
    resp = client.post('/analyze', data=form())
    assert resp.status_code == 200
    body = resp.data.decode()

    # Ana bölümlerin hepsi render olmalı
    for marker in ['Reviews Scraped', 'Complaints Filtered', 'Platform Comparison',
                   'Rating Distribution', 'Complaint Themes', 'Save as PDF']:
        assert marker in body, f'eksik bölüm: {marker}'
    assert 'unexpected error occurred' not in body


def test_analyze_shows_date_filter_badge(client, patched_fetch):
    resp = client.post('/analyze', data=form(start_date='2024-02-01'))
    body = resp.data.decode()
    assert 'Date filter active' in body
    assert '2024-02-01' in body


def test_analyze_without_dates_hides_badge(client, patched_fetch):
    resp = client.post('/analyze', data=form())
    assert 'Date filter active' not in resp.data.decode()


def test_analyze_reports_error_when_no_complaints(client, patched_fetch):
    resp = client.post('/analyze', data=form(complaint_threshold='0'))
    assert resp.status_code == 200
    assert 'No complaint reviews were found' in resp.data.decode()


def test_analyze_reports_error_when_no_reviews(client, monkeypatch):
    from reviewMetrix import analyzer
    monkeypatch.setattr(analyzer, 'fetch_reviews_store',
                        lambda *a: (pd.DataFrame(), {'google': None, 'ios': None}))

    resp = client.post('/analyze', data=form())
    assert resp.status_code == 200
    assert 'No reviews could be found' in resp.data.decode()


def test_analyze_handles_invalid_number_gracefully(client, patched_fetch):
    """Bozuk sayısal girdi 500 değil, hata sayfası döndürmeli."""
    resp = client.post('/analyze', data=form(max_reviews='not-a-number'))
    assert resp.status_code == 200
    assert 'unexpected error occurred' in resp.data.decode()


# --------------------------------------------------------------------------
# /compare
# --------------------------------------------------------------------------

def test_compare_renders_two_apps(client, patched_fetch):
    resp = client.post('/compare', data=form(
        google_id_b='com.mock.b', apple_name_b='mockb'
    ))
    assert resp.status_code == 200
    body = resp.data.decode()

    for marker in ['VS', 'Avg Store Rating', 'themesCompareChart',
                   'ratingCompareChart', 'sentimentCompareChart', 'Save as PDF']:
        assert marker in body, f'eksik bölüm: {marker}'


def test_compare_survives_one_failing_app(client, monkeypatch, reviews_df, summary_stats):
    """B uygulaması veri döndürmese bile sayfa render olmalı ve uyarı göstermeli."""
    from reviewMetrix import analyzer

    def selective_fetch(google_id, apple_name, country, lang, max_reviews):
        if google_id == 'com.mock.b':
            return pd.DataFrame(), {'google': None, 'ios': None}
        return reviews_df.copy(), summary_stats

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', selective_fetch)

    resp = client.post('/compare', data=form(
        google_id_b='com.mock.b', apple_name_b='mockb'
    ))
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'No reviews could be found' in body
    assert 'MockApp' in body, "çalışan uygulamanın verisi yine gösterilmeli"


# --------------------------------------------------------------------------
# /compare-countries
# --------------------------------------------------------------------------

def test_compare_countries_renders_all_countries(client, patched_fetch):
    resp = client.post('/compare-countries', data=form(countries='us, gb, de'))
    assert resp.status_code == 200
    body = resp.data.decode()

    for code in ['US', 'GB', 'DE']:
        assert f'class="country-code">{code}' in body
    assert 'Market comparison for' in body
    assert 'ratingByCountry' in body
    assert 'Save as PDF' in body


def test_compare_countries_requires_at_least_one_country(client, patched_fetch):
    resp = client.post('/compare-countries', data=form(countries='  ,  '))
    assert resp.status_code == 200
    assert 'at least one country code' in resp.data.decode()


def test_compare_countries_deduplicates_and_caps_at_six(client, patched_fetch):
    resp = client.post('/compare-countries', data=form(
        countries='us, us, gb, de, fr, tr, jp, it, es'
    ))
    body = resp.data.decode()

    # us tekrarı sayılmamalı, toplam 6 ülke ile sınırlanmalı
    assert body.count('class="country-code">') == 6
    assert 'class="country-code">US' in body
    assert body.count('class="country-code">US') == 1
    # 7. ve sonraki ülkeler dahil edilmemeli
    assert 'class="country-code">IT' not in body


def test_compare_countries_normalizes_case_and_whitespace(client, patched_fetch):
    resp = client.post('/compare-countries', data=form(countries='  US ,  gB '))
    body = resp.data.decode()
    assert 'class="country-code">US' in body
    assert 'class="country-code">GB' in body


# --------------------------------------------------------------------------
# _avg_store_rating yardımcı fonksiyonu
# --------------------------------------------------------------------------

@pytest.mark.parametrize('stats,expected', [
    ({'google': {'rating': 4.0}, 'ios': {'rating': 5.0}}, 4.5),
    ({'google': {'rating': 4.0}, 'ios': None}, 4.0),
    ({'google': None, 'ios': {'rating': 3.0}}, 3.0),
    ({'google': None, 'ios': None}, None),
    (None, None),
    ({'google': {'rating': 0}, 'ios': None}, None),  # 0 puan = veri yok sayılır
])
def test_avg_store_rating(stats, expected):
    assert _avg_store_rating(stats) == expected

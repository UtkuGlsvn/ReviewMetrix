"""Paylaşılan pytest fixture'ları.

Tüm testler ağ erişimi olmadan çalışır: mağaza scraping'i yapan
fetch_reviews_store fonksiyonu testlerde monkeypatch ile değiştirilir.
"""
import pandas as pd
import pytest

from reviewMetrix import create_app


@pytest.fixture(autouse=True)
def clear_cache():
    """Her testten önce/sonra scraping cache'ini boşaltır.

    Cache modül düzeyinde yaşadığı için temizlenmezse bir testin sahte verisi
    başka bir teste sızar ve monkeypatch'ler etkisiz kalır.
    """
    from reviewMetrix import analyzer
    analyzer.clear_review_cache()
    yield
    analyzer.clear_review_cache()


@pytest.fixture
def app():
    """Test için Flask uygulaması."""
    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def reviews_df():
    """İki platformdan, farklı puan/tarih/sürümlere sahip örnek yorum seti."""
    return pd.DataFrame({
        'review': [
            'App keeps crashing after the update',
            'Too many ads everywhere',
            'Cannot login to my account',
            'Great app, love it',
            'Very slow and laggy, battery drain',
            'Refund my money, subscription too expensive',
            'Crash on startup again',
            'Works fine for me',
        ],
        'score': [1, 2, 1, 5, 2, 1, 1, 4],
        'date': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
            '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04',
        ]),
        'platform': [
            'Google Play', 'App Store', 'Google Play', 'App Store',
            'Google Play', 'App Store', 'Google Play', 'App Store',
        ],
        'version': ['5.0', None, '5.0', None, '5.1', None, '5.1', None],
    })


@pytest.fixture
def summary_stats():
    # description_full ASO ve rakip listeleme analizinin kullandigi alandir;
    # gercek veride her zaman bulunur
    google_desc = 'A mock app for listening to music, podcasts and audiobooks offline.'
    ios_desc = 'A mock app with playlists, radio stations and offline downloads.'
    return {
        'google': {
            'rating': 4.2, 'reviews': 1000, 'title': 'MockApp', 'developer': 'MockDev',
            'category': 'Social', 'installs': '1B+', 'version': '5.1',
            'released': '2015', 'description': google_desc[:200],
            'description_full': google_desc,
        },
        'ios': {
            'rating': 4.6, 'reviews': 500, 'title': 'MockApp', 'developer': 'MockDev',
            'category': 'Social', 'version': '5.1', 'released': '2015-01-01',
            'description': ios_desc[:200], 'description_full': ios_desc,
        },
    }


@pytest.fixture
def patched_fetch(monkeypatch, reviews_df, summary_stats):
    """fetch_reviews_store'u ağ erişimi olmadan sabit veri döndürecek şekilde değiştirir."""
    from reviewMetrix import analyzer

    def fake_fetch(google_id, apple_name, country, lang, max_reviews):
        return reviews_df.copy(), summary_stats

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', fake_fetch)
    return fake_fetch

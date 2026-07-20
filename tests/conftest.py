"""Shared pytest fixtures.

Every test runs without network access: fetch_reviews_store, which does the
store scraping, is monkeypatched out.
"""
import pandas as pd
import pytest

from reviewMetrix import create_app


@pytest.fixture(autouse=True)
def clear_cache():
    """Empty the scraping cache before and after every test.

    The cache lives at module level, so without this one test's mock data
    leaks into another and the monkeypatches silently stop mattering.
    """
    from reviewMetrix import analyzer
    analyzer.clear_review_cache()
    yield
    analyzer.clear_review_cache()


@pytest.fixture
def app():
    """Flask app under test."""
    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def reviews_df():
    """Sample reviews from both platforms, across ratings, dates and versions."""
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
    # description_full is what the ASO and competitor listing analysis read;
    # real payloads always carry it
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
    """Replace fetch_reviews_store with fixed data, so no network is needed."""
    from reviewMetrix import analyzer

    def fake_fetch(google_id, apple_name, country, lang, max_reviews):
        return reviews_df.copy(), summary_stats

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', fake_fetch)
    return fake_fetch

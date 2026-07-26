"""Tests for accepting full store URLs in the app id / name fields."""
import pytest

from reviewMetrix import analyzer


# --------------------------------------------------------------------------
# parse_google_id
# --------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    ('com.google.android.youtube', 'com.google.android.youtube'),
    ('https://play.google.com/store/apps/details?id=com.google.android.youtube',
     'com.google.android.youtube'),
    ('https://play.google.com/store/apps/details?id=com.spotify.music&hl=en&gl=US',
     'com.spotify.music'),
    ('https://play.google.com/store/apps/details?hl=en&id=com.whatsapp',
     'com.whatsapp'),
    ('  com.instagram.android  ', 'com.instagram.android'),
    ('', ''),
    (None, ''),
])
def test_parse_google_id(value, expected):
    assert analyzer.parse_google_id(value) == expected


# --------------------------------------------------------------------------
# parse_apple_name
# --------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    ('youtube', 'youtube'),
    ('https://apps.apple.com/us/app/youtube/id544007664', 'youtube'),
    ('https://apps.apple.com/us/app/spotify-music-and-podcasts/id324684580?l=en',
     'spotify-music-and-podcasts'),
    ('https://apps.apple.com/tr/app/instagram/id389801252', 'instagram'),
    ('  netflix  ', 'netflix'),
    ('', ''),
    (None, ''),
])
def test_parse_apple_name(value, expected):
    assert analyzer.parse_apple_name(value) == expected


def test_apple_url_without_slug_left_unchanged():
    """A URL that carries only the numeric id has no slug to extract."""
    value = 'https://apps.apple.com/app/id544007664'
    assert analyzer.parse_apple_name(value) == value


# --------------------------------------------------------------------------
# Route integration: the scraper receives the parsed id/name, not the URL
# --------------------------------------------------------------------------

def test_analyze_parses_pasted_urls(client, monkeypatch, reviews_df, summary_stats):
    seen = {}

    def capture(google_id, apple_name, country, lang, max_reviews):
        seen['google_id'] = google_id
        seen['apple_name'] = apple_name
        return reviews_df.copy(), summary_stats

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', capture)

    resp = client.post('/analyze', data={
        'google_id': 'https://play.google.com/store/apps/details?id=com.spotify.music&hl=en',
        'apple_name': 'https://apps.apple.com/us/app/spotify-music-and-podcasts/id324684580',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })

    assert resp.status_code == 200
    assert seen['google_id'] == 'com.spotify.music'
    assert seen['apple_name'] == 'spotify-music-and-podcasts'


def test_compare_parses_competitor_urls(client, monkeypatch, reviews_df, summary_stats):
    seen = []

    def capture(google_id, apple_name, country, lang, max_reviews):
        seen.append(google_id)
        return reviews_df.copy(), summary_stats

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', capture)

    resp = client.post('/compare', data={
        'google_id': 'com.spotify.music',
        'apple_name': 'spotify-music',
        'google_id_b': 'https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically',
        'apple_name_b': 'tiktok',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })

    assert resp.status_code == 200
    assert 'com.zhiliaoapp.musically' in seen, 'the competitor URL should have been parsed'

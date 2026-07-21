"""Tests for the ASO (App Store Optimization) analysis."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


def _stats(title='Mock Music Player', description='Listen to music and podcasts. Stream playlists offline.',
           rating=4.3, reviews=5000):
    store = {
        'rating': rating, 'reviews': reviews, 'title': title, 'developer': 'Dev',
        'category': 'Music', 'installs': '1M+', 'version': '1.0',
        'released': '2020', 'description': description[:200],
        'description_full': description,
    }
    return {'google': store, 'ios': None}


def _reviews(texts):
    return pd.DataFrame({
        'review': texts,
        'score': [3] * len(texts),
        'date': pd.to_datetime(['2024-01-01'] * len(texts)),
        'platform': ['Google Play'] * len(texts),
    })


# --------------------------------------------------------------------------
# Metadata health
# --------------------------------------------------------------------------

def test_metadata_health_reports_lengths_and_limits():
    report = analyzer.get_aso_report(_stats(), _reviews(['great app']), 'en')
    meta = report['metadata'][0]

    assert meta['store'] == 'Google Play'
    assert meta['title_length'] == len('Mock Music Player')
    assert meta['title_limit'] == 30
    assert meta['description_limit'] == 4000


def test_short_title_flagged_as_wasted_keyword_space():
    """A title well under the limit wastes valuable keyword space."""
    short = analyzer.get_aso_report(_stats(title='App'), _reviews(['x']), 'en')
    full = analyzer.get_aso_report(_stats(title='Mock Music Player Radio Songs'), _reviews(['x']), 'en')

    assert short['metadata'][0]['title_status'] != 'good'
    assert full['metadata'][0]['title_status'] == 'good'


def test_rating_and_review_volume_status_thresholds():
    weak = analyzer.get_aso_report(_stats(rating=2.9, reviews=20), _reviews(['x']), 'en')['metadata'][0]
    strong = analyzer.get_aso_report(_stats(rating=4.6, reviews=50000), _reviews(['x']), 'en')['metadata'][0]

    assert weak['rating_status'] == 'bad'
    assert weak['reviews_status'] == 'bad'
    assert strong['rating_status'] == 'good'
    assert strong['reviews_status'] == 'good'


def test_metadata_covers_both_stores_when_present():
    stats = _stats()
    stats['ios'] = dict(stats['google'])
    report = analyzer.get_aso_report(stats, _reviews(['x']), 'en')

    assert {m['store'] for m in report['metadata']} == {'Google Play', 'App Store'}


# --------------------------------------------------------------------------
# Keyword gap, the valuable output
# --------------------------------------------------------------------------

def test_keyword_gap_surfaces_terms_missing_from_listing():
    """Users say "alarm" and the listing never does, so it is an opportunity."""
    stats = _stats(description='Listen to music and podcasts. Stream playlists offline.')
    reviews = _reviews([
        'I use the alarm every morning',
        'The alarm feature is great',
        'Please improve the alarm',
        'love the music',
    ])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    terms = [k['term'] for k in report['keyword_opportunities']]

    assert 'alarm' in terms, f'alarm not surfaced: {terms}'


def test_terms_already_in_listing_are_excluded():
    stats = _stats(description='Listen to music and podcasts every day.')
    reviews = _reviews(['music music music', 'podcasts are good', 'music again'])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    terms = [k['term'] for k in report['keyword_opportunities']]

    assert 'music' not in terms
    assert 'podcasts' not in terms


def test_single_mention_terms_are_ignored_as_noise():
    """One mention is noise, two is signal.

    The words are tagged as nouns in context, so this measures the frequency
    threshold rather than the noun filter.
    """
    stats = _stats(description='Music app.')
    reviews = _reviews([
        'the widget is broken',      # tek kez -> elenmeli
        'i need a timer',            # iki kez -> kalmali
        'i need a timer',
    ])

    terms = [k['term'] for k in analyzer.get_aso_report(stats, reviews, 'en')['keyword_opportunities']]
    assert 'timer' in terms, f'a twice-mentioned word should stay: {terms}'
    assert 'widget' not in terms, f'a single mention should be dropped: {terms}'


def test_phrases_are_detected_and_flagged():
    stats = _stats(description='A music app.')
    reviews = _reviews([
        'the sound quality is bad',
        'sound quality dropped',
        'sound quality again',
    ])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    phrases = [k['term'] for k in report['keyword_opportunities'] if k['is_phrase']]

    assert 'sound quality' in phrases
    for k in report['keyword_opportunities']:
        assert k['is_phrase'] == (' ' in k['term'])


def test_generic_filler_words_are_filtered_out():
    """Words like "would", "getting" and "thank" carry no ASO signal."""
    stats = _stats(description='A music app.')
    reviews = _reviews([
        'I would really like getting this thank you',
        'I would really like getting this thank you',
        'I would really like getting this thank you',
    ])

    terms = [k['term'] for k in analyzer.get_aso_report(stats, reviews, 'en')['keyword_opportunities']]
    for noise in ['would', 'getting', 'thank', 'really', 'like']:
        assert noise not in terms, f'noise word not filtered: {noise}'


def test_percent_is_share_of_reviews():
    stats = _stats(description='A music app.')
    reviews = _reviews(['alarm here', 'alarm there', 'nothing', 'nothing'])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    alarm = next(k for k in report['keyword_opportunities'] if k['term'] == 'alarm')
    assert alarm['mentions'] == 2
    assert alarm['percent'] == 50.0


# --------------------------------------------------------------------------
# Listing keywords
# --------------------------------------------------------------------------

def test_listing_keywords_reflect_description():
    stats = _stats(description='Music streaming with playlists. Music podcasts and music radio.')
    report = analyzer.get_aso_report(stats, _reviews(['x']), 'en')

    terms = [k['term'] for k in report['listing_keywords']]
    assert 'music' in terms
    top = report['listing_keywords'][0]
    assert top['term'] == 'music', 'the most frequent listing word should lead'


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_no_reviews_still_returns_metadata():
    report = analyzer.get_aso_report(_stats(), pd.DataFrame(), 'en')
    assert report['metadata']
    assert report['keyword_opportunities'] == []
    assert report['available'] is True


def test_no_stats_returns_unavailable():
    report = analyzer.get_aso_report(None, pd.DataFrame(), 'en')
    assert report['available'] is False
    assert report['metadata'] == []


def test_missing_description_does_not_crash():
    stats = _stats(description='')
    report = analyzer.get_aso_report(stats, _reviews(['alarm', 'alarm']), 'en')
    assert report['metadata'][0]['description_length'] == 0


def test_non_english_language_still_produces_gap():
    """The POS tagger is English only; other languages fall back to stopwords."""
    stats = _stats(description='Müzik dinleme uygulaması.')
    reviews = _reviews(['alarm özelliği çok iyi', 'alarm kurulumu kolay', 'alarm lazım'])

    report = analyzer.get_aso_report(stats, reviews, 'tr')
    terms = [k['term'] for k in report['keyword_opportunities']]
    assert 'alarm' in terms


def test_noun_filter_degrades_gracefully_without_tagger(monkeypatch):
    """Without a tagger the tokens come back unchanged, with no exception."""
    import nltk
    monkeypatch.setattr(nltk, 'pos_tag', lambda *a, **k: (_ for _ in ()).throw(LookupError('no tagger')))

    tokens = ['alarm', 'clock', 'would']
    assert analyzer._noun_filter(tokens, 'en') == tokens


def test_noun_filter_skipped_for_non_english():
    tokens = ['alarm', 'özellik']
    assert analyzer._noun_filter(tokens, 'tr') == tokens


# --------------------------------------------------------------------------
# Pipeline and route integration
# --------------------------------------------------------------------------

def test_build_app_report_includes_aso(monkeypatch):
    stats = _stats(description='Music streaming app.')
    reviews = _reviews(['alarm feature', 'alarm again', 'music good'])
    monkeypatch.setattr(analyzer, 'fetch_reviews_store', lambda *a: (reviews.copy(), stats))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50, 3, 10)
    assert report['aso']['available'] is True
    assert report['aso']['metadata']


def test_aso_computed_even_when_no_complaints(monkeypatch):
    """ASO ignores the complaint filter."""
    stats = _stats()
    reviews = _reviews(['alarm feature', 'alarm again'])  # hepsi 3 puan
    monkeypatch.setattr(analyzer, 'fetch_reviews_store', lambda *a: (reviews.copy(), stats))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50,
                                       complaint_threshold=1, top_words=10)
    assert report['error'] is not None, 'no complaints expected'
    assert report['aso']['available'] is True, 'ASO should still be computed'


def test_results_page_renders_aso_section(client, patched_fetch):
    resp = client.post('/analyze', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })
    body = resp.data.decode()

    assert resp.status_code == 200
    assert 'ASO — Store Listing Optimization' in body
    assert 'Keyword opportunities' in body
    # ASO lives in its own tab
    assert 'data-tab="aso"' in body
    assert 'data-panel="aso"' in body


# --------------------------------------------------------------------------
# Store creatives (icon, screenshots)
# --------------------------------------------------------------------------

def test_metadata_reports_screenshot_count_and_icon():
    stats = _stats()
    stats['google'].update({
        'icon': 'https://example.test/icon.png',
        'screenshots': ['https://example.test/1.png', 'https://example.test/2.png'],
        'screenshot_count': 6,
        'store_url': 'https://play.google.com/store/apps/details?id=x',
    })
    meta = analyzer.get_aso_report(stats, _reviews(['x']), 'en')['metadata'][0]

    assert meta['screenshots'] == 6
    assert meta['screenshots_status'] == 'good'
    assert meta['icon'] == 'https://example.test/icon.png'
    assert len(meta['shots']) == 2
    assert meta['store_url'].startswith('https://play.google.com')


@pytest.mark.parametrize('count,expected', [
    (0, 'bad'), (1, 'bad'), (2, 'warn'), (3, 'warn'), (4, 'good'), (40, 'good'),
])
def test_screenshot_status_thresholds(count, expected):
    """A listing with one or two screenshots is under-using merchandising space."""
    stats = _stats()
    stats['google']['screenshot_count'] = count
    meta = analyzer.get_aso_report(stats, _reviews(['x']), 'en')['metadata'][0]
    assert meta['screenshots_status'] == expected


def test_metadata_handles_missing_creatives():
    """Older cached payloads carry no image fields; that must not crash."""
    meta = analyzer.get_aso_report(_stats(), _reviews(['x']), 'en')['metadata'][0]
    assert meta['screenshots'] == 0
    assert meta['icon'] is None
    assert meta['shots'] == []


def test_listing_strength_reports_screenshot_count():
    stats = {
        'google': {'title': 'A', 'description_full': 'x', 'rating': 4.0,
                   'reviews': 10, 'screenshot_count': 3},
        'ios': {'title': 'A', 'description_full': 'x', 'rating': 4.0,
                'reviews': 10, 'screenshot_count': 8},
    }
    assert analyzer.get_listing_strength(stats)['screenshots'] == 8

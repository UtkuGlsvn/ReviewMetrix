"""ASO (App Store Optimization) analiz testleri."""
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
# Metadata sağlığı
# --------------------------------------------------------------------------

def test_metadata_health_reports_lengths_and_limits():
    report = analyzer.get_aso_report(_stats(), _reviews(['great app']), 'en')
    meta = report['metadata'][0]

    assert meta['store'] == 'Google Play'
    assert meta['title_length'] == len('Mock Music Player')
    assert meta['title_limit'] == 30
    assert meta['description_limit'] == 4000


def test_short_title_flagged_as_wasted_keyword_space():
    """Limitin çok altındaki başlık değerli keyword alanını boşa harcar."""
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
# Keyword gap — asıl değer
# --------------------------------------------------------------------------

def test_keyword_gap_surfaces_terms_missing_from_listing():
    """Kullanıcılar 'alarm' diyor, listeleme hiç bahsetmiyor -> fırsat."""
    stats = _stats(description='Listen to music and podcasts. Stream playlists offline.')
    reviews = _reviews([
        'I use the alarm every morning',
        'The alarm feature is great',
        'Please improve the alarm',
        'love the music',
    ])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    terms = [k['term'] for k in report['keyword_opportunities']]

    assert 'alarm' in terms, f'alarm bulunamadi: {terms}'


def test_terms_already_in_listing_are_excluded():
    stats = _stats(description='Listen to music and podcasts every day.')
    reviews = _reviews(['music music music', 'podcasts are good', 'music again'])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    terms = [k['term'] for k in report['keyword_opportunities']]

    assert 'music' not in terms
    assert 'podcasts' not in terms


def test_single_mention_terms_are_ignored_as_noise():
    """Bir kez gecen kelime gurultudur; iki kez gecen sinyaldir.

    Kullanilan kelimeler cumle icinde isim olarak etiketlenir, boylece testin
    frekans esigini olctugunden emin oluruz (isim filtresini degil).
    """
    stats = _stats(description='Music app.')
    reviews = _reviews([
        'the widget is broken',      # tek kez -> elenmeli
        'i need a timer',            # iki kez -> kalmali
        'i need a timer',
    ])

    terms = [k['term'] for k in analyzer.get_aso_report(stats, reviews, 'en')['keyword_opportunities']]
    assert 'timer' in terms, f'iki kez gecen kelime kalmali: {terms}'
    assert 'widget' not in terms, f'tek seferlik kelime elenmeli: {terms}'


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
    """'would', 'getting', 'thank' gibi kelimeler ASO sinyali degildir."""
    stats = _stats(description='A music app.')
    reviews = _reviews([
        'I would really like getting this thank you',
        'I would really like getting this thank you',
        'I would really like getting this thank you',
    ])

    terms = [k['term'] for k in analyzer.get_aso_report(stats, reviews, 'en')['keyword_opportunities']]
    for noise in ['would', 'getting', 'thank', 'really', 'like']:
        assert noise not in terms, f'gurultu kelimesi elenmemis: {noise}'


def test_percent_is_share_of_reviews():
    stats = _stats(description='A music app.')
    reviews = _reviews(['alarm here', 'alarm there', 'nothing', 'nothing'])

    report = analyzer.get_aso_report(stats, reviews, 'en')
    alarm = next(k for k in report['keyword_opportunities'] if k['term'] == 'alarm')
    assert alarm['mentions'] == 2
    assert alarm['percent'] == 50.0


# --------------------------------------------------------------------------
# Listeleme kelimeleri
# --------------------------------------------------------------------------

def test_listing_keywords_reflect_description():
    stats = _stats(description='Music streaming with playlists. Music podcasts and music radio.')
    report = analyzer.get_aso_report(stats, _reviews(['x']), 'en')

    terms = [k['term'] for k in report['listing_keywords']]
    assert 'music' in terms
    top = report['listing_keywords'][0]
    assert top['term'] == 'music', 'en sik gecen listeleme kelimesi basta olmali'


# --------------------------------------------------------------------------
# Kenar durumlar
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
    """POS etiketleyici yalnizca Ingilizce; diger dillerde stopword filtresine duser."""
    stats = _stats(description='Müzik dinleme uygulaması.')
    reviews = _reviews(['alarm özelliği çok iyi', 'alarm kurulumu kolay', 'alarm lazım'])

    report = analyzer.get_aso_report(stats, reviews, 'tr')
    terms = [k['term'] for k in report['keyword_opportunities']]
    assert 'alarm' in terms


def test_noun_filter_degrades_gracefully_without_tagger(monkeypatch):
    """Etiketleyici yoksa token'lar oldugu gibi donmeli, hata firlamamali."""
    import nltk
    monkeypatch.setattr(nltk, 'pos_tag', lambda *a, **k: (_ for _ in ()).throw(LookupError('no tagger')))

    tokens = ['alarm', 'clock', 'would']
    assert analyzer._noun_filter(tokens, 'en') == tokens


def test_noun_filter_skipped_for_non_english():
    tokens = ['alarm', 'özellik']
    assert analyzer._noun_filter(tokens, 'tr') == tokens


# --------------------------------------------------------------------------
# Pipeline ve route entegrasyonu
# --------------------------------------------------------------------------

def test_build_app_report_includes_aso(monkeypatch):
    stats = _stats(description='Music streaming app.')
    reviews = _reviews(['alarm feature', 'alarm again', 'music good'])
    monkeypatch.setattr(analyzer, 'fetch_reviews_store', lambda *a: (reviews.copy(), stats))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50, 3, 10)
    assert report['aso']['available'] is True
    assert report['aso']['metadata']


def test_aso_computed_even_when_no_complaints(monkeypatch):
    """ASO sikayet filtresinden bagimsizdir."""
    stats = _stats()
    reviews = _reviews(['alarm feature', 'alarm again'])  # hepsi 3 puan
    monkeypatch.setattr(analyzer, 'fetch_reviews_store', lambda *a: (reviews.copy(), stats))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50,
                                       complaint_threshold=1, top_words=10)
    assert report['error'] is not None, 'sikayet bulunmamali'
    assert report['aso']['available'] is True, 'ASO yine de hesaplanmali'


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
    # ASO kendi sekmesinde yer alir
    assert 'data-tab="aso"' in body
    assert 'data-panel="aso"' in body

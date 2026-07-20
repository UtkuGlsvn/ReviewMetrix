"""Tests for multilingual sentiment and theme detection."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


# --------------------------------------------------------------------------
# Language support flag
# --------------------------------------------------------------------------

@pytest.mark.parametrize('lang', ['en', 'tr', 'de', 'es', 'fr', 'EN', 'Tr'])
def test_supported_languages(lang):
    assert analyzer.is_sentiment_supported(lang) is True


@pytest.mark.parametrize('lang', ['ja', 'ru', 'zh', 'ar'])
def test_unsupported_languages(lang):
    assert analyzer.is_sentiment_supported(lang) is False


@pytest.mark.parametrize('lang', ['', None])
def test_missing_language_defaults_to_english(lang):
    """With no language the pipeline assumes English, consistent with
    get_sentiment and get_theme_keywords."""
    assert analyzer.is_sentiment_supported(lang) is True


# --------------------------------------------------------------------------
# Sentiment: negative reviews must score negative
# --------------------------------------------------------------------------

@pytest.mark.parametrize('lang,text', [
    ('tr', 'Uygulama sürekli çöküyor ve çok yavaş. Berbat!'),
    ('de', 'Die App stürzt ständig ab und ist sehr langsam. Schrecklich!'),
    ('es', 'La aplicación es terrible y muy lenta.'),
    ('fr', 'Application horrible, très lente.'),
    ('en', 'The app keeps crashing and is very slow. Terrible!'),
])
def test_negative_reviews_score_negative(lang, text):
    score = analyzer.get_sentiment(text, lang)
    assert score < -0.2, f'{lang}: expected negative, got {score}'


@pytest.mark.parametrize('lang,text', [
    ('tr', 'Harika bir uygulama, mükemmel!'),
    ('de', 'Super App, großartig und schnell!'),
    ('es', 'Excelente aplicación, muy útil.'),
    ('fr', 'Application excellente et rapide.'),
    ('en', 'Wonderful app, absolutely great!'),
])
def test_positive_reviews_score_positive(lang, text):
    score = analyzer.get_sentiment(text, lang)
    assert score > 0.2, f'{lang}: expected positive, got {score}'


def test_regression_non_english_no_longer_silently_neutral():
    """Old behaviour: TextBlob returned 0.0 for every non-English review."""
    harsh_turkish = 'Uygulama berbat, sürekli çöküyor!'
    assert analyzer.get_sentiment(harsh_turkish, 'tr') < -0.2
    # The TextBlob path still returns 0.0, which was the root of the problem
    assert analyzer.get_sentiment(harsh_turkish, 'en') == 0.0


def test_unsupported_language_returns_zero():
    assert analyzer.get_sentiment('これはひどいアプリです', 'ja') == 0.0


@pytest.mark.parametrize('bad', [None, '', '   ', 123])
def test_invalid_input_returns_zero(bad):
    assert analyzer.get_sentiment(bad, 'tr') == 0.0


def test_sentiment_is_bounded():
    extreme = 'berbat rezalet korkunç iğrenç felaket ' * 10
    assert -1.0 <= analyzer.get_sentiment(extreme, 'tr') <= 1.0


# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('lang,text', [
    ('tr', 'Uygulama hiç iyi değil'),
    ('de', 'Die App ist nicht gut'),
    ('fr', "Ce n est pas bon"),
])
def test_negation_flips_polarity(lang, text):
    """"iyi"/"gut"/"bon" are positive; negation must flip them."""
    assert analyzer.get_sentiment(text, lang) < 0


def test_negation_only_applies_within_window():
    """Negation must not reach a distant word."""
    near = analyzer.get_sentiment('nicht gut', 'de')
    far = analyzer.get_sentiment('nicht ist das wirklich sehr gut', 'de')
    assert near < 0
    assert far > 0


# --------------------------------------------------------------------------
# Multilingual theme detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize('lang,text,expected', [
    ('tr', 'Uygulama sürekli çöküyor ve çok yavaş', {'Crashes & Bugs', 'Performance'}),
    ('tr', 'Çok fazla reklam var, giriş yapamıyorum', {'Ads', 'Login & Account'}),
    ('de', 'Die App stürzt ab und ist langsam', {'Crashes & Bugs', 'Performance'}),
    ('de', 'Zu viele Werbung, ich kann mich nicht anmelden', {'Ads', 'Login & Account'}),
    ('es', 'La aplicación es lenta y tiene muchos anuncios', {'Performance', 'Ads'}),
    ('fr', 'Application lente et trop chère', {'Performance', 'Price & Payment'}),
    ('en', 'App keeps crashing and is slow', {'Crashes & Bugs', 'Performance'}),
])
def test_themes_detected_per_language(lang, text, expected):
    df = pd.DataFrame({'review': [text]})
    found = {t['theme'] for t in analyzer.categorize_complaints(df, lang)}
    assert expected <= found, f'{lang}: expected {expected}, found {found}'


def test_regression_non_english_themes_were_missed():
    """Old behaviour: no theme was ever detected in a non-English review."""
    df = pd.DataFrame({'review': ['Uygulama sürekli çöküyor ve çok yavaş']})
    assert analyzer.categorize_complaints(df, 'en') == []      # the old path
    assert analyzer.categorize_complaints(df, 'tr') != []      # the fixed path


def test_english_keywords_still_matched_in_other_languages():
    """Reviews often mix in English terms, so both lists must be searched."""
    df = pd.DataFrame({'review': ['Uygulamada sürekli crash var']})
    found = {t['theme'] for t in analyzer.categorize_complaints(df, 'tr')}
    assert 'Crashes & Bugs' in found


def test_get_theme_keywords_merges_language_with_english():
    tr = analyzer.get_theme_keywords('tr')
    assert 'crash' in tr['Crashes & Bugs']      # English
    assert 'çöküyor' in tr['Crashes & Bugs']    # Turkish

    en = analyzer.get_theme_keywords('en')
    assert 'çöküyor' not in en['Crashes & Bugs']


def test_get_theme_keywords_unknown_language_falls_back_to_english():
    ja = analyzer.get_theme_keywords('ja')
    assert ja['Crashes & Bugs'] == analyzer.COMPLAINT_THEMES['Crashes & Bugs']


# --------------------------------------------------------------------------
# Pipeline integration
# --------------------------------------------------------------------------

def _turkish_df():
    return pd.DataFrame({
        'review': [
            'Uygulama sürekli çöküyor, berbat',
            'Çok yavaş ve reklam dolu',
            'Giriş yapamıyorum, şifre hatası',
            'Harika uygulama',
            'Çok pahalı bir abonelik',
            'Bildirimler çalışmıyor',
        ],
        'score': [1, 2, 1, 5, 2, 1],
        'date': pd.to_datetime(['2024-01-0%d' % i for i in range(1, 7)]),
        'platform': ['Google Play'] * 6,
        'version': ['5.0'] * 6,
    })


def test_pipeline_uses_language_for_sentiment_and_themes():
    df = _turkish_df()
    complaints = analyzer.preprocess_and_filter_complaints(df, 2, 'mockapp', 'tr')

    # Turkish sentiment should have been computed, not left at 0.0
    assert (complaints['sentiment'] != 0).any(), 'Turkish sentiment was not computed'

    result = analyzer.analyze_and_visualize(complaints, top_n=10)
    assert result['sentiment_available'] is True
    assert result['lang_code'] == 'tr'
    assert result['themes'], 'Turkish themes should be found'


def test_pipeline_flags_unsupported_language():
    df = _turkish_df()
    complaints = analyzer.preprocess_and_filter_complaints(df, 2, 'mockapp', 'ja')
    result = analyzer.analyze_and_visualize(complaints, top_n=10)

    assert result['sentiment_available'] is False
    assert result['lang_code'] == 'ja'


def test_analyze_lang_code_falls_back_to_preprocess_value():
    """Without an explicit language, the one attached during preprocessing wins."""
    df = _turkish_df()
    complaints = analyzer.preprocess_and_filter_complaints(df, 2, 'mockapp', 'de')
    result = analyzer.analyze_and_visualize(complaints, top_n=10)  # lang_code verilmedi
    assert result['lang_code'] == 'de'


def test_build_app_report_propagates_language(monkeypatch):
    df = _turkish_df()
    monkeypatch.setattr(analyzer, 'fetch_reviews_store',
                        lambda *a: (df.copy(), {'google': None, 'ios': None}))

    report = analyzer.build_app_report('com.x', 'x', 'tr', 'tr', 50, 2, 10)
    assert report['sentiment_available'] is True
    assert report['lang_code'] == 'tr'
    assert report['themes']


def test_build_app_report_unsupported_language_flag(monkeypatch):
    df = _turkish_df()
    monkeypatch.setattr(analyzer, 'fetch_reviews_store',
                        lambda *a: (df.copy(), {'google': None, 'ios': None}))

    report = analyzer.build_app_report('com.x', 'x', 'jp', 'ja', 50, 2, 10)
    assert report['sentiment_available'] is False


# --------------------------------------------------------------------------
# UI honesty: sentiment sections hide for unsupported languages
# --------------------------------------------------------------------------

def _analyze_form(**over):
    data = {
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    }
    data.update(over)
    return data


def test_unsupported_language_shows_notice_and_hides_sentiment(client, patched_fetch):
    resp = client.post('/analyze', data=_analyze_form(language='ja'))
    body = resp.data.decode()

    assert resp.status_code == 200
    assert 'Sentiment analysis unavailable' in body
    assert '<h3>Sentiment Analysis</h3>' not in body
    assert 'Average Sentiment Polarity' not in body


def test_supported_language_shows_sentiment_sections(client, patched_fetch):
    resp = client.post('/analyze', data=_analyze_form(language='en'))
    body = resp.data.decode()

    assert 'Sentiment analysis unavailable' not in body
    assert '<h3>Sentiment Analysis</h3>' in body

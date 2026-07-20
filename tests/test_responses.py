"""Geliştirici yanıt analizi ve rakip ASO listeleme boşluğu testleri."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


def _df(rows, platform='Google Play'):
    """rows: (review, score, reply, hours_to_reply) dortlulerinden DataFrame."""
    posted = pd.Timestamp('2024-01-01 09:00:00')
    return pd.DataFrame({
        'review': [r[0] for r in rows],
        'score': [r[1] for r in rows],
        'reply': [r[2] for r in rows],
        'replied_at': [
            (posted + pd.Timedelta(hours=r[3])) if r[2] and r[3] is not None else None
            for r in rows
        ],
        'date': [posted] * len(rows),
        'platform': [platform] * len(rows),
        'likes': [0] * len(rows),
    })


def _stats(title='Mock Music', description='Listen to music and podcasts offline.',
           developer='MockDev'):
    return {
        'google': {
            'rating': 4.2, 'reviews': 100, 'title': title, 'developer': developer,
            'category': 'Music', 'installs': '1M', 'version': '1', 'released': '2020',
            'description': description[:200], 'description_full': description,
        },
        'ios': None,
    }


# ==========================================================================
# Yanıt oranı ve hızı
# ==========================================================================

def test_response_rate_counts_replies():
    df = _df([
        ('crash', 1, 'Sorry to hear that', 2),
        ('ads', 2, None, None),
        ('ads again', 2, None, None),
        ('great', 5, 'Thanks!', 1),
    ])
    r = analyzer.get_response_analysis(df)

    assert r['available'] is True
    assert r['total'] == 4
    assert r['replied'] == 2
    assert r['response_rate'] == 50.0


def test_complaint_response_rate_is_measured_separately():
    """Sikayetlere yanit orani, genel orandan daha anlamli bir gostergedir."""
    df = _df([
        ('crash', 1, None, None),      # sikayet, yanitsiz
        ('ads', 2, None, None),        # sikayet, yanitsiz
        ('love it', 5, 'Thanks!', 1),  # ovgu, yanitli
        ('nice', 5, 'Thanks!', 1),     # ovgu, yanitli
    ])
    r = analyzer.get_response_analysis(df)

    assert r['response_rate'] == 50.0, 'genel oran iyi gorunuyor'
    assert r['complaints_total'] == 2
    assert r['complaints_replied'] == 0
    assert r['complaint_response_rate'] == 0.0, 'ama sikayetlerin hicbirine yanit yok'


def test_median_reply_time_in_hours():
    df = _df([
        ('a', 1, 'reply', 2),
        ('b', 1, 'reply', 4),
        ('c', 1, 'reply', 12),
    ])
    assert analyzer.get_response_analysis(df)['median_hours'] == 4.0


def test_negative_reply_times_are_ignored():
    """Bozuk veri (yanit tarihi yorumdan once) medyani bozmamali."""
    df = _df([('a', 1, 'reply', -50), ('b', 1, 'reply', 6)])
    assert analyzer.get_response_analysis(df)['median_hours'] == 6.0


def test_blank_replies_do_not_count():
    df = _df([('a', 1, '   ', None), ('b', 1, None, None)])
    r = analyzer.get_response_analysis(df)
    assert r['replied'] == 0
    assert r['response_rate'] == 0.0


# ==========================================================================
# Tema bazlı ihmal analizi — asıl aksiyon alınabilir çıktı
# ==========================================================================

def test_by_theme_surfaces_ignored_themes_first():
    df = _df([
        # Ads: 3 sikayet, hicbirine yanit yok
        ('too many ads', 2, None, None),
        ('ads everywhere', 2, None, None),
        ('more ads', 2, None, None),
        # Crashes: 2 sikayet, ikisine de yanit
        ('app crash', 1, 'We are on it', 3),
        ('crash again', 1, 'Please contact us', 4),
    ])
    by_theme = {t['theme']: t for t in analyzer.get_response_analysis(df)['by_theme']}

    assert by_theme['Ads']['rate'] == 0.0
    assert by_theme['Crashes & Bugs']['rate'] == 100.0
    # En dusuk oranli tema listede once gelmeli
    first = analyzer.get_response_analysis(df)['by_theme'][0]
    assert first['theme'] == 'Ads'


def test_by_theme_excludes_themes_with_no_reviews():
    df = _df([('too many ads', 2, None, None)])
    themes = [t['theme'] for t in analyzer.get_response_analysis(df)['by_theme']]
    assert 'Ads' in themes
    assert 'Notifications' not in themes


# ==========================================================================
# Kenar durumlar
# ==========================================================================

def test_app_store_only_returns_unavailable():
    """Yanit verisi yalnizca Google Play'de var."""
    df = _df([('crash', 1, None, None)], platform='App Store')
    assert analyzer.get_response_analysis(df)['available'] is False


def test_only_google_play_rows_are_measured():
    df = pd.concat([
        _df([('a', 1, 'reply', 2)], platform='Google Play'),
        _df([('b', 1, None, None)], platform='App Store'),
    ], ignore_index=True)
    r = analyzer.get_response_analysis(df)

    assert r['total'] == 1, 'App Store satirlari orani seyreltmemeli'
    assert r['response_rate'] == 100.0


def test_missing_reply_column_returns_unavailable():
    df = pd.DataFrame({'review': ['x'], 'score': [1]})
    assert analyzer.get_response_analysis(df)['available'] is False


def test_empty_dataframe_returns_unavailable():
    assert analyzer.get_response_analysis(pd.DataFrame())['available'] is False


def test_no_replies_at_all_still_available():
    """Hic yanit yoksa da bu bir bulgudur: oran %0."""
    df = _df([('a', 1, None, None), ('b', 2, None, None)])
    r = analyzer.get_response_analysis(df)
    assert r['available'] is True
    assert r['response_rate'] == 0.0
    assert r['median_hours'] is None


def test_build_app_report_includes_responses(monkeypatch):
    df = _df([('crash', 1, 'Sorry!', 3), ('ads', 2, None, None)])
    monkeypatch.setattr(analyzer, 'fetch_reviews_store', lambda *a: (df.copy(), _stats()))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50, 2, 10)
    assert report['responses']['available'] is True
    assert report['responses']['response_rate'] == 50.0


# ==========================================================================
# Rakip ASO listeleme boşluğu
# ==========================================================================

def test_competitor_gap_splits_unique_keywords():
    a = _stats(title='App A', description='Listen to podcasts and audiobooks offline.')
    b = _stats(title='App B', description='Watch videos and live performances.')

    gap = analyzer.get_competitor_keyword_gap(a, b)
    a_terms = [k['term'] for k in gap['a_only']]
    b_terms = [k['term'] for k in gap['b_only']]

    assert gap['available'] is True
    assert 'podcasts' in a_terms
    assert 'performances' in b_terms
    assert 'podcasts' not in b_terms


def test_competitor_gap_excludes_shared_terms():
    a = _stats(title='App A', description='Music streaming with playlists.')
    b = _stats(title='App B', description='Music streaming with radio.')

    gap = analyzer.get_competitor_keyword_gap(a, b)
    assert 'music' not in [k['term'] for k in gap['a_only']]
    assert 'music' not in [k['term'] for k in gap['b_only']]
    assert gap['shared'] > 0


def test_competitor_gap_excludes_brand_names():
    """Rakibin marka adi hedeflenebilir bir ASO kelimesi degildir."""
    a = _stats(title='Spotify Music', description='Spotify brings you music.',
               developer='Spotify AB')
    b = _stats(title='Deezer Audio', description='Deezer brings you radio.',
               developer='Deezer')

    gap = analyzer.get_competitor_keyword_gap(a, b)
    all_terms = [k['term'] for k in gap['a_only'] + gap['b_only']]

    assert 'spotify' not in all_terms
    assert 'deezer' not in all_terms


def test_competitor_gap_unavailable_without_listings():
    assert analyzer.get_competitor_keyword_gap(None, None)['available'] is False


def test_competitor_gap_respects_top_n():
    long_desc = ' '.join(f'keyword{i} feature' for i in range(40))
    a = _stats(description=long_desc)
    b = _stats(description='Nothing in common here.')

    gap = analyzer.get_competitor_keyword_gap(a, b, top_n=5)
    assert len(gap['a_only']) <= 5


# ==========================================================================
# POS etiketleme bağlam düzeltmesi
# ==========================================================================

def test_noun_filter_uses_raw_context_when_given():
    """Stopword'leri atilmis bir yiginda POS etiketleyici baglamsiz kalir ve yanilir.

    Olculen kontrast: ayni token listesi icin 'wherever' baglamsiz yiginda isim
    (NN) sayilip tutuluyor, ham cumle verildiginde zarf (WRB) oldugu anlasilip
    dogru sekilde eleniyor.
    """
    raw = 'listen to music and podcasts and audiobooks wherever you are'
    tokens = ['music', 'podcasts', 'audiobooks', 'wherever']

    with_context = analyzer._noun_filter(tokens, 'en', context_text=raw)
    without_context = analyzer._noun_filter(tokens, 'en')

    assert 'wherever' not in with_context, 'baglamla zarf oldugu anlasilmali'
    assert 'wherever' in without_context, 'baglamsiz halde yanlis etiketlenir'
    # Gercek isimler her iki durumda da korunur
    assert {'music', 'podcasts'} <= set(with_context)


def test_noun_filter_non_english_returns_tokens_unchanged():
    tokens = ['müzik', 'podcast']
    assert analyzer._noun_filter(tokens, 'tr', context_text='müzik ve podcast') == tokens


def test_noun_set_returns_none_without_tagger(monkeypatch):
    import nltk
    monkeypatch.setattr(nltk, 'pos_tag', lambda *a, **k: (_ for _ in ()).throw(LookupError()))
    assert analyzer._noun_set('some text', 'en') is None


# ==========================================================================
# Route entegrasyonu
# ==========================================================================

def test_results_page_renders_response_section(client, patched_fetch):
    resp = client.post('/analyze', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })
    body = resp.data.decode()
    assert resp.status_code == 200
    # patched_fetch yanit kolonu uretmiyor -> bolum gizli olmali
    assert ('Developer Responses' in body) == ('href="#responses"' in body)


def test_compare_page_renders_keyword_gap(client, monkeypatch, reviews_df):
    """Iki uygulamanin listelemeleri FARKLI olmali; ayni listelemede bosluk
    cikmamasi dogru davranistir."""
    def fetch_by_app(google_id, apple_name, country, lang, max_reviews):
        if google_id == 'com.mock.b':
            return reviews_df.copy(), _stats(
                title='Rival Video', description='Watch videos and live performances.',
                developer='RivalDev')
        return reviews_df.copy(), _stats(
            title='Mock Music', description='Listen to podcasts and audiobooks offline.')

    monkeypatch.setattr(analyzer, 'fetch_reviews_store', fetch_by_app)

    resp = client.post('/compare', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'google_id_b': 'com.mock.b', 'apple_name_b': 'mockb',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })
    body = resp.data.decode()

    assert resp.status_code == 200
    assert 'ASO Listing Gap' in body
    assert 'performances' in body, 'rakibin hedefledigi kelime gosterilmeli'


def test_compare_identical_listings_show_no_gap(client, patched_fetch):
    """Ayni listelemeye sahip iki uygulama arasinda ASO boslugu olmamali."""
    resp = client.post('/compare', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'google_id_b': 'com.mock.b', 'apple_name_b': 'mockb',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })
    assert resp.status_code == 200
    assert 'ASO Listing Gap' not in resp.data.decode()

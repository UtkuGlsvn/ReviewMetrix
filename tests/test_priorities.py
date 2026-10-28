"""Tests for momentum, issue prioritization and likes weighting."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


def _df(rows):
    """rows: (review, score, likes) uclulerinden DataFrame uretir."""
    return pd.DataFrame({
        'review': [r[0] for r in rows],
        'score': [r[1] for r in rows],
        'likes': [r[2] for r in rows],
        'date': pd.to_datetime(['2024-01-0%d' % ((i % 9) + 1) for i in range(len(rows))]),
        'platform': ['Google Play'] * len(rows),
    })


def _stats(google_rating=4.5, ios_rating=None):
    stats = {'google': None, 'ios': None}
    if google_rating is not None:
        stats['google'] = {
            'rating': google_rating, 'reviews': 1000, 'title': 'MockApp',
            'developer': 'Dev', 'category': 'x', 'installs': '1M', 'version': '1',
            'released': '2020', 'description': 'd', 'description_full': 'd',
        }
    if ios_rating is not None:
        stats['ios'] = {
            'rating': ios_rating, 'reviews': 500, 'title': 'MockApp',
            'developer': 'Dev', 'category': 'x', 'version': '1',
            'released': '2020-01-01', 'description': 'd', 'description_full': 'd',
        }
    return stats


# ==========================================================================
# 1. Momentum — yasam boyu vs guncel puan
# ==========================================================================

def test_momentum_detects_decline():
    """Yasam boyu 4.5, guncel yorumlar 2.0 -> dususte."""
    df = _df([('bad', 2, 0), ('bad', 2, 0), ('bad', 2, 0)])
    result = analyzer.get_momentum(_stats(google_rating=4.5), df)

    assert result['available'] is True
    p = result['platforms'][0]
    assert p['lifetime'] == 4.5
    assert p['recent'] == 2.0
    assert p['delta'] == -2.5
    assert p['status'] == 'declining'
    assert p['sample'] == 3


def test_momentum_detects_improvement():
    df = _df([('good', 5, 0), ('good', 5, 0)])
    p = analyzer.get_momentum(_stats(google_rating=3.0), df)['platforms'][0]

    assert p['delta'] == 2.0
    assert p['status'] == 'improving'


def test_momentum_reports_stable_within_threshold():
    """Kucuk farklar gurultu sayilir, 'dususte' denmez."""
    df = _df([('ok', 4, 0), ('ok', 4, 0)])
    p = analyzer.get_momentum(_stats(google_rating=4.1), df)['platforms'][0]

    assert p['status'] == 'stable'
    assert abs(p['delta']) < 0.3


@pytest.mark.parametrize('lifetime,recent_score,expected', [
    (4.5, 2, 'declining'),
    (4.0, 4, 'stable'),
    (3.0, 5, 'improving'),
    (4.29, 4, 'stable'),      # -0.29: esigin hemen icinde -> henuz dusus sayilmaz
    (4.31, 4, 'declining'),   # -0.31: esigin hemen otesinde -> dusus
])
def test_momentum_status_thresholds(lifetime, recent_score, expected):
    df = _df([('x', recent_score, 0)] * 3)
    assert analyzer.get_momentum(_stats(google_rating=lifetime), df)['platforms'][0]['status'] == expected


def test_momentum_uses_only_matching_platform_reviews():
    """Her platformun guncel ortalamasi kendi yorumlarindan hesaplanmali."""
    df = pd.DataFrame({
        'review': ['a', 'b', 'c', 'd'],
        'score': [1, 1, 5, 5],
        'likes': [0, 0, 0, 0],
        'date': pd.to_datetime(['2024-01-01'] * 4),
        'platform': ['Google Play', 'Google Play', 'App Store', 'App Store'],
    })
    result = analyzer.get_momentum(_stats(google_rating=4.0, ios_rating=4.0), df)
    by_platform = {p['platform']: p for p in result['platforms']}

    assert by_platform['Google Play']['recent'] == 1.0
    assert by_platform['App Store']['recent'] == 5.0
    assert by_platform['Google Play']['status'] == 'declining'
    assert by_platform['App Store']['status'] == 'improving'


def test_momentum_skips_platform_without_store_rating():
    df = _df([('x', 3, 0)])
    result = analyzer.get_momentum(_stats(google_rating=4.0, ios_rating=None), df)
    assert [p['platform'] for p in result['platforms']] == ['Google Play']


def test_momentum_unavailable_without_reviews():
    assert analyzer.get_momentum(_stats(), pd.DataFrame())['available'] is False


def test_momentum_unavailable_without_stats():
    assert analyzer.get_momentum(None, _df([('x', 3, 0)]))['available'] is False


# ==========================================================================
# 3 + 4. Tema onceliklendirme ve begeni agirliklandirma
# ==========================================================================

def test_theme_priority_outranks_raw_volume():
    """Az sikayet ama cok begeni + agir puan, cok sikayetli hafif temayi gecmeli.

    Bu ozelligin can alici davranisi: siralamanin ham adetten farkli olmasi.
    """
    rows = [
        # Ads: 4 sikayet, begeni yok, 2 yildiz (daha hafif)
        ('too many ads', 2, 0), ('ads everywhere', 2, 0),
        ('ads again', 2, 0), ('more ads', 2, 0),
        # Privacy: 1 sikayet ama 50 begeni ve 1 yildiz (agir)
        ('privacy scam, my data was leaked', 1, 50),
    ]
    themes = analyzer.categorize_complaints(_df(rows))
    by_theme = {t['theme']: t for t in themes}

    assert by_theme['Ads']['count'] > by_theme['Privacy & Ads Tracking']['count'], \
        'Ads ham adette onde olmali'
    assert themes[0]['theme'] == 'Privacy & Ads Tracking', \
        f"begeni agirlikli oncelik kazanmali, siralama: {[t['theme'] for t in themes]}"


def test_likes_count_as_affected_users():
    rows = [('too many ads', 2, 9)]
    t = analyzer.categorize_complaints(_df(rows))[0]

    assert t['count'] == 1
    assert t['likes'] == 9
    assert t['affected'] == 10, 'etkilenen = sikayet + begeni'


def test_severity_raises_priority_for_lower_ratings():
    """Ayni adet ve begeni, daha dusuk puan -> daha yuksek oncelik."""
    harsh = analyzer.categorize_complaints(_df([('crash', 1, 0)]))[0]
    mild = analyzer.categorize_complaints(_df([('crash', 4, 0)]))[0]

    assert harsh['priority_raw'] > mild['priority_raw']


def test_theme_reports_average_score():
    rows = [('crash', 1, 0), ('crash bug', 3, 0)]
    t = analyzer.categorize_complaints(_df(rows))[0]
    assert t['avg_score'] == 2.0


def test_priority_normalized_zero_to_hundred():
    rows = [('ads', 2, 10), ('crash', 1, 0), ('login problem', 2, 1)]
    themes = analyzer.categorize_complaints(_df(rows))

    assert themes[0]['priority'] == 100, 'en yuksek tema 100 olmali'
    for t in themes:
        assert 0 <= t['priority'] <= 100


def test_categorize_works_without_likes_column():
    """Sadece App Store sonuclarinda begeni kolonu bulunmaz."""
    df = pd.DataFrame({
        'review': ['too many ads', 'crash on login'],
        'score': [2, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
    })
    themes = analyzer.categorize_complaints(df)

    assert themes
    assert all(t['likes'] == 0 for t in themes)
    assert all(t['affected'] == t['count'] for t in themes)


def test_categorize_handles_missing_score_column():
    df = pd.DataFrame({'review': ['too many ads', 'crash']})
    themes = analyzer.categorize_complaints(df)
    assert themes
    assert all(t['avg_score'] is None for t in themes)


# ==========================================================================
# Ornek yorumlar begeniye gore siralanir
# ==========================================================================

def test_sample_reviews_sorted_by_likes():
    rows = [
        ('barely noticed complaint', 1, 0),
        ('the one everyone agrees with', 1, 99),
        ('mildly liked', 1, 5),
    ]
    df = _df(rows)
    complaints = analyzer.preprocess_and_filter_complaints(df, 2, 'mockapp', 'en')
    result = analyzer.analyze_and_visualize(complaints, top_n=5)

    likes = [r['likes'] for r in result['sample_reviews']]
    assert likes == sorted(likes, reverse=True), 'en cok begenilen ustte olmali'
    assert result['sample_reviews'][0]['likes'] == 99


# ==========================================================================
# Pipeline ve route entegrasyonu
# ==========================================================================

def test_build_app_report_includes_momentum(monkeypatch):
    df = _df([('bad app', 1, 3)] * 4)
    monkeypatch.setattr(analyzer, 'fetch_reviews_store',
                        lambda *a: (df.copy(), _stats(google_rating=4.6)))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50, 2, 10)
    assert report['momentum']['available'] is True
    assert report['momentum']['platforms'][0]['status'] == 'declining'


def test_momentum_computed_even_without_complaints(monkeypatch):
    """Momentum sikayet filtresinden bagimsizdir."""
    df = _df([('great', 5, 0)] * 3)
    monkeypatch.setattr(analyzer, 'fetch_reviews_store',
                        lambda *a: (df.copy(), _stats(google_rating=4.0)))

    report = analyzer.build_app_report('com.x', 'x', 'us', 'en', 50,
                                       complaint_threshold=1, top_words=10)
    assert report['error'] is not None, 'sikayet bulunmamali'
    assert report['momentum']['available'] is True


def test_results_page_renders_momentum_and_priorities(client, patched_fetch):
    resp = client.post('/analyze', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })
    body = resp.data.decode()

    assert resp.status_code == 200
    # Momentum genel bakis sekmesinde, oncelik tablosu ise Issues sekmesinde
    assert 'Rating Momentum' in body
    assert 'data-panel="overview"' in body
    assert 'Fix First' in body
    assert '30 days before the newest review' in body
    assert 'Users affected' in body
    assert 'data-tab="issues"' in body


# ==========================================================================
# Yenilik agirligi — duzeltilmis eski bir hata ust sirada kalmamali
# ==========================================================================

def test_fixed_old_bug_drops_below_recent_issue():
    """Eski ama cok sayida crash sikayeti, yeni ama daha az login sikayetinin
    gerisine dusmeli; ham adette yine onde gorunmeli."""
    old = pd.Timestamp('2024-01-01')
    new = pd.Timestamp('2024-06-01')
    df = pd.DataFrame({
        'review': ['app crash'] * 6 + ['login problem'] * 3,
        'score': [1] * 9,
        'likes': [0] * 9,
        'date': [old] * 6 + [new] * 3,
        'platform': ['Google Play'] * 9,
    })
    themes = analyzer.categorize_complaints(df)
    by_theme = {t['theme']: t for t in themes}

    crash = next(t for name, t in by_theme.items() if 'Crash' in name)
    login = next(t for name, t in by_theme.items() if 'Login' in name)
    assert crash['count'] > login['count']
    assert crash['affected'] == 6, 'tablo ham etkilenen sayisini gostermeli'
    assert themes[0] is login, f"siralama: {[t['theme'] for t in themes]}"


def test_recency_weight_halves_per_half_life(monkeypatch):
    monkeypatch.setattr(analyzer, 'PRIORITY_HALF_LIFE_DAYS', 10.0)
    work = pd.DataFrame({'date': pd.to_datetime(['2024-01-21', '2024-01-11', '2024-01-01'])})
    weights = analyzer._recency_weights(work).tolist()
    assert weights == pytest.approx([1.0, 0.5, 0.25])


def test_recency_disabled_or_undated_gives_equal_weight(monkeypatch):
    undated = pd.DataFrame({'date': [None, 'not a date']})
    assert analyzer._recency_weights(undated).tolist() == [1.0, 1.0]

    monkeypatch.setattr(analyzer, 'PRIORITY_HALF_LIFE_DAYS', 0)
    dated = pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2023-01-01'])})
    assert analyzer._recency_weights(dated).tolist() == [1.0, 1.0]


# ==========================================================================
# Log olcekli begeni ve surum farkindaligi
# ==========================================================================

def _versioned(rows, platform='Google Play'):
    """rows: (review, likes, version, days_ago) dortlulerinden DataFrame uretir."""
    now = pd.Timestamp('2024-06-30')
    return pd.DataFrame({
        'review': [r[0] for r in rows],
        'score': [1] * len(rows),
        'likes': [r[1] for r in rows],
        'version': [r[2] for r in rows],
        'date': [now - pd.Timedelta(days=r[3]) for r in rows],
        'platform': [platform] * len(rows),
    })


def _theme(themes, word):
    return next(t for t in themes if word in t['theme'])


def test_likes_weigh_on_log_scale():
    """Bir yorumun 500 begenisi, 500 ayri sikayet kadar agir sayilmamali."""
    df = _df([('app crash', 1, 500)] + [('login problem', 1, 0)] * 10)
    themes = analyzer.categorize_complaints(df)

    assert themes[0]['theme'].startswith('Login'), [t['theme'] for t in themes]
    assert _theme(themes, 'Crash')['affected'] == 501, 'tablo ham sayiyi gostermeli'


def test_viral_old_crash_drops_below_new_login_issue():
    """Analizdeki senaryo: eski surumde 500 begenili crash, yeni surumde 10
    login sikayeti. Crash son surumde yok -> muhtemelen duzeldi."""
    df = _versioned([('app crash', 500, '3.1', 30)]
                    + [('login problem', 0, '4.0', d % 3) for d in range(10)])
    themes = analyzer.categorize_complaints(df)

    assert themes[0]['theme'].startswith('Login')
    assert _theme(themes, 'Crash')['version_status'] == 'likely_fixed'
    assert _theme(themes, 'Login')['version_status'] == 'new'


def test_theme_missing_from_latest_version_is_demoted():
    rows = ([('app crash', 0, '1.0', 5)] * 10
            + [('too many ads', 0, '1.0', 5)] * 10
            + [('too many ads', 0, '1.1', 0)] * 10)
    themes = analyzer.categorize_complaints(_versioned(rows))
    crash = _theme(themes, 'Crash')

    assert crash['version_status'] == 'likely_fixed'
    # Ayni tarih ve adette surum bilgisi olmasa oncelikler esit olurdu
    undated = analyzer.categorize_complaints(_versioned(rows).drop(columns='version'))
    assert crash['priority_raw'] == pytest.approx(
        _theme(undated, 'Crash')['priority_raw'] * analyzer.PRIORITY_FIXED_FACTOR, abs=0.01)


def test_theme_growing_on_latest_version_is_boosted():
    # Eski surum: 20 sikayetin 2'si login (%10). Yeni surum: 10'un 5'i (%50).
    rows = ([('login problem', 0, '2.0', 10)] * 2 + [('too many ads', 0, '2.0', 10)] * 18
            + [('login problem', 0, '2.1', 0)] * 5 + [('too many ads', 0, '2.1', 0)] * 5)
    themes = analyzer.categorize_complaints(_versioned(rows))
    login = _theme(themes, 'Login')

    assert login['version_status'] == 'rising'
    plain = analyzer.categorize_complaints(_versioned(rows).drop(columns='version'))
    assert login['priority_raw'] == pytest.approx(
        _theme(plain, 'Login')['priority_raw'] * analyzer.PRIORITY_MAX_BOOST, abs=0.01)
    assert _theme(themes, 'Ads')['version_status'] is None


def test_small_latest_version_sample_is_not_judged():
    """Yeni surumde birkac yorum varken eski temalar 'duzeldi' sayilmamali."""
    rows = [('app crash', 0, '1.0', 5)] * 10 + [('too many ads', 0, '1.1', 0)] * 3
    themes = analyzer.categorize_complaints(_versioned(rows))
    assert all(t['version_status'] is None for t in themes)


def test_versions_compared_per_platform():
    """iOS ve Android surum numaralari bagimsizdir; Android'in 9.0'i iOS'un
    2.0'ini 'eski' yapmamali."""
    android = _versioned([('app crash', 0, '8.0', 5)] * 10 + [('too many ads', 0, '9.0', 0)] * 10)
    ios = _versioned([('app crash', 0, '2.0', 0)] * 10, platform='App Store')
    themes = analyzer.categorize_complaints(pd.concat([android, ios], ignore_index=True))

    assert _theme(themes, 'Crash')['version_status'] is None, \
        'crash iOS son surumunde hala var'


def test_version_key_orders_numerically():
    assert analyzer._version_key('4.10.0') > analyzer._version_key('4.9.3')
    assert analyzer._version_key('4.0') == analyzer._version_key('4.0.0')
    assert analyzer._version_key(None) is None
    assert analyzer._version_key('beta') is None


def test_results_page_shows_version_badge(client, monkeypatch):
    rows = [('app crash', 0, '1.0', 5)] * 10 + [('too many ads', 0, '1.1', 0)] * 10
    df = _versioned(rows)
    monkeypatch.setattr(analyzer, 'fetch_reviews_store',
                        lambda *a: (df.copy(), _stats(google_rating=4.0)))
    resp = client.post('/analyze', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10', 'refresh': '1',
    })
    assert resp.status_code == 200
    assert 'LIKELY FIXED' in resp.data.decode()

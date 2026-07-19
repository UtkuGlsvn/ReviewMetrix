"""Momentum, tema önceliklendirme ve beğeni ağırlıklandırma testleri."""
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
    assert 'Rating Momentum' in body
    assert 'href="#momentum"' in body
    assert 'Fix First' in body
    assert 'Users affected' in body

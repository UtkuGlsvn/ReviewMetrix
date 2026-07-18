"""analyzer.py içindeki saf analiz fonksiyonlarının testleri (ağ erişimi yok)."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


# --------------------------------------------------------------------------
# get_rating_distribution
# --------------------------------------------------------------------------

def test_rating_distribution_counts_overall_and_per_platform(reviews_df):
    dist = analyzer.get_rating_distribution(reviews_df)

    # scores: 1,2,1,5,2,1,1,4  -> dört adet 1, iki adet 2, bir adet 4, bir adet 5
    assert dist['overall'] == {'1': 4, '2': 2, '3': 0, '4': 1, '5': 1}
    assert set(dist['by_platform']) == {'Google Play', 'App Store'}
    # Her platformun toplamı kendi yorum sayısına eşit olmalı
    assert sum(dist['by_platform']['Google Play'].values()) == 4
    assert sum(dist['by_platform']['App Store'].values()) == 4


def test_rating_distribution_handles_empty_and_missing_columns():
    assert analyzer.get_rating_distribution(pd.DataFrame())['overall'] == {str(s): 0 for s in range(1, 6)}
    assert analyzer.get_rating_distribution(None)['by_platform'] == {}
    no_score = pd.DataFrame({'review': ['x']})
    assert analyzer.get_rating_distribution(no_score)['overall'] == {str(s): 0 for s in range(1, 6)}


# --------------------------------------------------------------------------
# categorize_complaints
# --------------------------------------------------------------------------

def test_categorize_complaints_detects_expected_themes(reviews_df):
    themes = analyzer.categorize_complaints(reviews_df)
    found = {t['theme'] for t in themes}

    assert 'Crashes & Bugs' in found      # "crashing", "crash on startup"
    assert 'Ads' in found                 # "too many ads"
    assert 'Login & Account' in found     # "cannot login"
    assert 'Performance' in found         # "slow and laggy, battery drain"
    assert 'Price & Payment' in found     # "refund my money, subscription"


def test_categorize_complaints_sorted_desc_and_percent_bounded(reviews_df):
    themes = analyzer.categorize_complaints(reviews_df)

    counts = [t['count'] for t in themes]
    assert counts == sorted(counts, reverse=True), "temalar azalan sırada olmalı"
    for t in themes:
        assert t['count'] > 0
        assert 0 < t['percent'] <= 100


def test_categorize_complaints_empty_input():
    assert analyzer.categorize_complaints(pd.DataFrame()) == []
    assert analyzer.categorize_complaints(None) == []


# --------------------------------------------------------------------------
# get_platform_comparison
# --------------------------------------------------------------------------

def test_platform_comparison_reports_both_platforms(reviews_df):
    comparison = analyzer.get_platform_comparison(reviews_df)

    assert len(comparison) == 2
    by_platform = {c['platform']: c for c in comparison}
    assert by_platform['Google Play']['count'] == 4
    assert by_platform['App Store']['count'] == 4
    for row in comparison:
        assert 1 <= row['avg_score'] <= 5
        assert -1 <= row['avg_sentiment'] <= 1


def test_platform_comparison_without_platform_column(reviews_df):
    df = reviews_df.drop(columns=['platform'])
    assert analyzer.get_platform_comparison(df) == []


# --------------------------------------------------------------------------
# filter_by_date_range
# --------------------------------------------------------------------------

@pytest.mark.parametrize('start,end,expected', [
    (None, None, 8),              # filtre yok -> hepsi
    ('2024-02-01', None, 4),      # sadece başlangıç
    (None, '2024-01-31', 4),      # sadece bitiş
    ('2024-01-03', '2024-02-01', 3),
    ('2030-01-01', None, 0),      # aralık dışında
])
def test_filter_by_date_range(reviews_df, start, end, expected):
    result = analyzer.filter_by_date_range(reviews_df, start, end)
    assert len(result) == expected


def test_filter_by_date_range_includes_whole_end_day(reviews_df):
    """Bitiş günü tam olarak dahil edilmeli (gün ortasındaki saatler kesilmemeli)."""
    df = reviews_df.copy()
    df.loc[0, 'date'] = pd.Timestamp('2024-01-01 23:30:00')
    result = analyzer.filter_by_date_range(df, '2024-01-01', '2024-01-01')
    assert len(result) == 1


def test_filter_by_date_range_handles_timezone_aware_dates(reviews_df):
    df = reviews_df.copy()
    df['date'] = df['date'].dt.tz_localize('UTC')
    assert len(analyzer.filter_by_date_range(df, '2024-02-01', None)) == 4


def test_filter_by_date_range_returns_input_when_no_date_column():
    df = pd.DataFrame({'review': ['a'], 'score': [1]})
    assert len(analyzer.filter_by_date_range(df, '2024-01-01', None)) == 1


# --------------------------------------------------------------------------
# get_emerging_keywords
# --------------------------------------------------------------------------

def _emerging_df():
    """İlk yarıda 'ads' baskın; ikinci yarıda 'crash' (4/4) ve 'lag' (2/4) yükseliyor.

    İki yükselen kelimenin artış oranı kasıtlı olarak farklı (crash > lag), böylece
    sıralamanın yönü gerçekten test edilebiliyor.
    """
    return pd.DataFrame({
        'cleaned_review': [
            'ads spam', 'ads annoying', 'ads everywhere', 'payment issue',
            'crash lag', 'crash lag', 'crash startup', 'crash again',
        ],
        'date': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
            '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04',
        ]),
        'score': [1] * 8,
    })


def test_emerging_keywords_detects_rising_word():
    result = analyzer.get_emerging_keywords(_emerging_df())
    words = [k['word'] for k in result['keywords']]

    assert words[0] == 'crash', "en çok yükselen kelime ilk sırada olmalı"
    top = result['keywords'][0]
    assert top['is_new'] is True, "eski pencerede yoksa NEW işaretlenmeli"
    assert top['change'] > 0
    assert result['recent_count'] == 4
    assert result['previous_count'] == 4
    assert result['split_date'] == '2024-02-01'


def test_emerging_keywords_sorted_by_change_desc():
    """Yükseliş miktarına göre azalan sırada olmalı: crash (%100) > lag (%50)."""
    result = analyzer.get_emerging_keywords(_emerging_df())
    words = [k['word'] for k in result['keywords']]
    changes = [k['change'] for k in result['keywords']]

    assert words[:2] == ['crash', 'lag'], f'beklenen sıra crash, lag; gelen: {words}'
    assert changes == sorted(changes, reverse=True)
    assert changes[0] > changes[1], "farklı artış oranları ayırt edilmeli"


def test_emerging_keywords_excludes_declining_words():
    """Azalan kelimeler (ads) sonuçta yer almamalı."""
    result = analyzer.get_emerging_keywords(_emerging_df())
    assert 'ads' not in [k['word'] for k in result['keywords']]


def test_emerging_keywords_needs_minimum_data():
    small = _emerging_df().head(4)
    assert analyzer.get_emerging_keywords(small)['keywords'] == []


def test_emerging_keywords_missing_columns():
    df = pd.DataFrame({'review': ['x'] * 10})
    assert analyzer.get_emerging_keywords(df)['keywords'] == []


# --------------------------------------------------------------------------
# get_version_breakdown
# --------------------------------------------------------------------------

def test_version_breakdown_groups_by_version(reviews_df):
    breakdown = analyzer.get_version_breakdown(reviews_df)

    assert {v['version'] for v in breakdown} == {'5.0', '5.1'}
    assert all(v['complaints'] == 2 for v in breakdown)
    for v in breakdown:
        assert 1 <= v['avg_score'] <= 5
        assert -1 <= v['avg_sentiment'] <= 1


def test_version_breakdown_sorted_by_complaints_desc():
    """En çok şikayet alan sürüm ilk sırada olmalı (5.1 -> 3, 5.0 -> 2, 4.9 -> 1)."""
    df = pd.DataFrame({
        'review': ['crash', 'bug', 'freeze', 'ads', 'slow', 'login'],
        'score': [1, 1, 1, 2, 2, 1],
        'date': pd.to_datetime(['2024-01-0%d' % i for i in range(1, 7)]),
        'version': ['5.1', '5.1', '5.1', '5.0', '5.0', '4.9'],
    })

    breakdown = analyzer.get_version_breakdown(df)
    assert [v['version'] for v in breakdown] == ['5.1', '5.0', '4.9']
    assert [v['complaints'] for v in breakdown] == [3, 2, 1]


def test_version_breakdown_respects_top_n():
    df = pd.DataFrame({
        'review': ['x'] * 10,
        'score': [1] * 10,
        'date': pd.to_datetime(['2024-01-01'] * 10),
        'version': [f'{i}.0' for i in range(10)],
    })
    assert len(analyzer.get_version_breakdown(df, top_n=3)) == 3


def test_version_breakdown_empty_when_no_versions(reviews_df):
    """iOS-only sonuçlarda sürüm bilgisi yoktur -> bölüm boş dönmeli."""
    df = reviews_df.copy()
    df['version'] = None
    assert analyzer.get_version_breakdown(df) == []


def test_version_breakdown_missing_column(reviews_df):
    assert analyzer.get_version_breakdown(reviews_df.drop(columns=['version'])) == []


# --------------------------------------------------------------------------
# get_sentiment / preprocess_and_filter_complaints
# --------------------------------------------------------------------------

def test_get_sentiment_polarity_direction():
    assert analyzer.get_sentiment('This is wonderful and great') > 0
    assert analyzer.get_sentiment('This is terrible and awful') < 0


@pytest.mark.parametrize('bad', [None, '', '   ', 123])
def test_get_sentiment_handles_invalid_input(bad):
    assert analyzer.get_sentiment(bad) == 0.0


def test_preprocess_filters_by_threshold_and_cleans_text(reviews_df):
    complaints = analyzer.preprocess_and_filter_complaints(
        reviews_df, threshold=2, app_name='mockapp', lang_code='en'
    )

    # score <= 2 olan 6 yorum var
    assert len(complaints) == 6
    assert (complaints['score'] <= 2).all()
    assert 'sentiment' in complaints.columns
    assert 'cleaned_review' in complaints.columns
    # temizlenmiş metinde noktalama ve rakam kalmamalı
    joined = ' '.join(complaints['cleaned_review'])
    assert ',' not in joined and '.' not in joined
    assert not any(ch.isdigit() for ch in joined)


def test_preprocess_applies_extra_stopwords(reviews_df):
    complaints = analyzer.preprocess_and_filter_complaints(
        reviews_df, threshold=2, app_name='mockapp', lang_code='en',
        extra_stopwords_str='crashing, ads'
    )
    joined = ' '.join(complaints['cleaned_review'])
    assert 'crashing' not in joined
    assert 'ads' not in joined


def test_preprocess_returns_none_when_no_complaints(reviews_df):
    """Eşiğin altında yorum yoksa None dönmeli."""
    high_scores = reviews_df[reviews_df['score'] >= 4]
    assert analyzer.preprocess_and_filter_complaints(
        high_scores, threshold=2, app_name='x', lang_code='en'
    ) is None


def test_preprocess_returns_none_without_score_column():
    df = pd.DataFrame({'review': ['x']})
    assert analyzer.preprocess_and_filter_complaints(df, 2, 'x', 'en') is None


# --------------------------------------------------------------------------
# analyze_and_visualize
# --------------------------------------------------------------------------

def test_analyze_and_visualize_returns_all_expected_keys(reviews_df):
    complaints = analyzer.preprocess_and_filter_complaints(reviews_df, 2, 'mockapp', 'en')
    result = analyzer.analyze_and_visualize(complaints, top_n=10)

    expected = {
        'most_common_words', 'image_data', 'sentiment_summary', 'trend_data',
        'sample_reviews', 'themes', 'emerging_keywords', 'version_breakdown',
    }
    assert expected <= set(result)

    # sentiment özeti tutarlı olmalı
    s = result['sentiment_summary']
    assert s['positive'] + s['neutral'] + s['negative'] == len(complaints)
    # word cloud base64 üretilmiş olmalı
    assert result['image_data'] and isinstance(result['image_data'], str)
    # örnek yorumlar platform bilgisini taşımalı
    assert 'platform' in result['sample_reviews'][0]


def test_analyze_and_visualize_trend_data_is_json_safe(reviews_df):
    """trend_data içindeki değerler numpy değil, saf Python tipleri olmalı."""
    import json

    complaints = analyzer.preprocess_and_filter_complaints(reviews_df, 2, 'mockapp', 'en')
    result = analyzer.analyze_and_visualize(complaints, top_n=5)

    json.dumps(result['trend_data'])  # numpy tipi varsa burada patlar
    for row in result['trend_data']:
        assert isinstance(row['avg_score'], float)
        assert isinstance(row['count'], int)


# --------------------------------------------------------------------------
# build_app_report
# --------------------------------------------------------------------------

def test_build_app_report_happy_path(patched_fetch):
    report = analyzer.build_app_report(
        'com.mock', 'mock', 'us', 'en', max_reviews=50,
        complaint_threshold=2, top_words=10,
    )

    assert report['error'] is None
    assert report['name'] == 'MockApp', "mağaza başlığı görünen ad olmalı"
    assert report['total_reviews'] == 8
    assert report['complaint_count'] == 6
    assert report['rating_distribution'] is not None
    assert report['themes']
    assert report['version_breakdown']


def test_build_app_report_applies_date_filter(patched_fetch):
    report = analyzer.build_app_report(
        'com.mock', 'mock', 'us', 'en', 50, 2, 10,
        start_date='2024-02-01',
    )
    assert report['total_reviews'] == 4


def test_build_app_report_error_when_date_range_excludes_everything(patched_fetch):
    report = analyzer.build_app_report(
        'com.mock', 'mock', 'us', 'en', 50, 2, 10,
        start_date='2030-01-01',
    )
    assert report['error'] is not None
    assert 'date range' in report['error']
    assert report['total_reviews'] == 0


def test_build_app_report_error_when_no_reviews(monkeypatch):
    from reviewMetrix import analyzer as a
    monkeypatch.setattr(a, 'fetch_reviews_store',
                        lambda *args: (pd.DataFrame(), {'google': None, 'ios': None}))

    report = a.build_app_report('x', 'y', 'us', 'en', 50, 2, 10)
    assert report['error'] is not None
    assert report['total_reviews'] == 0


def test_build_app_report_error_when_no_complaints(patched_fetch):
    """Eşik çok düşükse şikayet bulunamaz ama özet veriler yine dönmeli."""
    report = analyzer.build_app_report(
        'com.mock', 'mock', 'us', 'en', 50,
        complaint_threshold=0, top_words=10,
    )
    assert report['error'] is not None
    assert report['total_reviews'] == 8, "şikayet yoksa da toplam yorum sayısı dolu olmalı"
    assert report['rating_distribution'] is not None

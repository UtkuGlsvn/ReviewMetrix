"""Tests for the pure analysis functions in analyzer.py; no network access."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


# --------------------------------------------------------------------------
# get_rating_distribution
# --------------------------------------------------------------------------

def test_rating_distribution_counts_overall_and_per_platform(reviews_df):
    dist = analyzer.get_rating_distribution(reviews_df)

    # scores 1,2,1,5,2,1,1,4 -> four 1s, two 2s, one 4, one 5
    assert dist['overall'] == {'1': 4, '2': 2, '3': 0, '4': 1, '5': 1}
    assert set(dist['by_platform']) == {'Google Play', 'App Store'}
    # Each platform's counts should sum to its own review total
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


def test_categorize_complaints_sorted_by_priority_and_percent_bounded(reviews_df):
    """Themes are ordered by priority, not by raw count."""
    themes = analyzer.categorize_complaints(reviews_df)

    priorities = [t['priority_raw'] for t in themes]
    assert priorities == sorted(priorities, reverse=True), "themes should be ordered by descending priority"
    for t in themes:
        assert t['count'] > 0
        assert 0 < t['percent'] <= 100
        assert 0 <= t['priority'] <= 100


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
    (None, None, 8),              # no filter -> everything
    ('2024-02-01', None, 4),      # start only
    (None, '2024-01-31', 4),      # end only
    ('2024-01-03', '2024-02-01', 3),
    ('2030-01-01', None, 0),      # outside the range
])
def test_filter_by_date_range(reviews_df, start, end, expected):
    result = analyzer.filter_by_date_range(reviews_df, start, end)
    assert len(result) == expected


def test_filter_by_date_range_includes_whole_end_day(reviews_df):
    """The end day is inclusive; times during that day must not be cut off."""
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
    """"ads" dominates the first half; "crash" (4/4) and "lag" (2/4) rise in
    the second.

    The two rising words climb by deliberately different amounts (crash > lag),
    so the direction of the sort is actually tested.
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

    assert words[0] == 'crash', "the fastest-rising word should lead"
    top = result['keywords'][0]
    assert top['is_new'] is True, "absent from the older window means NEW"
    assert top['change'] > 0
    assert result['recent_count'] == 4
    assert result['previous_count'] == 4
    assert result['split_date'] == '2024-02-01'


def test_emerging_keywords_sorted_by_change_desc():
    """Ordered by descending increase: crash (100%) > lag (50%)."""
    result = analyzer.get_emerging_keywords(_emerging_df())
    words = [k['word'] for k in result['keywords']]
    changes = [k['change'] for k in result['keywords']]

    assert words[:2] == ['crash', 'lag'], f'expected crash, lag; got: {words}'
    assert changes == sorted(changes, reverse=True)
    assert changes[0] > changes[1], "different rates of increase must be distinguished"


def test_emerging_keywords_excludes_declining_words():
    """Declining words such as "ads" must not appear."""
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
    """The version with the most complaints leads (5.1 -> 3, 5.0 -> 2, 4.9 -> 1)."""
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
    """iOS-only results carry no version, so the breakdown comes back empty."""
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
    # No punctuation or digits should survive cleaning
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
    """Returns None when nothing falls under the threshold."""
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

    # The sentiment summary should add up
    s = result['sentiment_summary']
    assert s['positive'] + s['neutral'] + s['negative'] == len(complaints)
    # A base64 word cloud should have been produced
    assert result['image_data'] and isinstance(result['image_data'], str)
    # Sample reviews should carry their platform
    assert 'platform' in result['sample_reviews'][0]


def test_analyze_and_visualize_trend_data_is_json_safe(reviews_df):
    """trend_data values must be plain Python types, not numpy ones."""
    import json

    complaints = analyzer.preprocess_and_filter_complaints(reviews_df, 2, 'mockapp', 'en')
    result = analyzer.analyze_and_visualize(complaints, top_n=5)

    json.dumps(result['trend_data'])  # a numpy type would blow up here
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
    assert report['name'] == 'MockApp', "the store title should be the display name"
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
    """Too low a threshold finds no complaints, but summary data still comes back."""
    report = analyzer.build_app_report(
        'com.mock', 'mock', 'us', 'en', 50,
        complaint_threshold=0, top_words=10,
    )
    assert report['error'] is not None
    assert report['total_reviews'] == 8, "the review total should be populated even with no complaints"
    assert report['rating_distribution'] is not None

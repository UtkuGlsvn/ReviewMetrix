"""Tests for the Average Rating card: daily averages, period deltas and
release markers."""
import pandas as pd
import pytest

from reviewMetrix import analyzer


def _df(rows):
    """rows: (date, score, version, platform) dortlulerinden DataFrame uretir."""
    return pd.DataFrame({
        'review': ['x'] * len(rows),
        'score': [r[1] for r in rows],
        'date': pd.to_datetime([r[0] for r in rows]),
        'version': [r[2] for r in rows],
        'platform': [r[3] for r in rows],
    })


GP, IOS = 'Google Play', 'App Store'


def test_daily_average_with_gaps_on_days_without_reviews():
    df = _df([('2024-03-01 09:00', 5, '1.0', GP), ('2024-03-01 18:00', 3, '1.0', GP),
              ('2024-03-04 10:00', 2, '1.0', GP)])
    w = analyzer.get_rating_timeline(df, windows=(7,))['series'][GP]['7']

    assert w['days'][0] == '2024-02-27' and w['days'][-1] == '2024-03-04'
    assert w['avg'] == [None, None, None, 4.0, None, None, 2.0]
    assert w['count'] == [0, 0, 0, 2, 0, 0, 1]


def test_window_average_is_weighted_by_reviews_not_days():
    df = _df([('2024-03-01', 5, '1.0', GP)] * 3 + [('2024-03-02', 1, '1.0', GP)])
    w = analyzer.get_rating_timeline(df, windows=(7,))['series'][GP]['7']
    assert w['average'] == 4.0       # (5+5+5+1)/4, gunlerin ortalamasi olsa 3.0 olurdu
    assert w['reviews'] == 4


def test_delta_against_previous_period():
    df = _df([('2024-03-01', 5, '1.0', GP), ('2024-03-02', 4, '1.0', GP),   # onceki 7 gun
              ('2024-03-09', 3, '1.0', GP), ('2024-03-14', 4, '1.0', GP)])  # son 7 gun
    w = analyzer.get_rating_timeline(df, windows=(7,))['series'][GP]['7']

    assert w['previous_average'] == 4.5
    assert w['average'] == 3.5
    assert w['delta'] == -1.0


def test_delta_is_none_without_previous_reviews():
    df = _df([('2024-03-10', 4, '1.0', GP)])
    w = analyzer.get_rating_timeline(df, windows=(7,))['series'][GP]['7']
    assert w['delta'] is None and w['previous_average'] is None


def test_platforms_are_split_and_all_combines_them():
    df = _df([('2024-03-10', 5, '1.0', GP), ('2024-03-10', 1, '2.0', IOS)])
    r = analyzer.get_rating_timeline(df, windows=(7,))

    assert r['platforms'] == ['All', GP, IOS]
    assert r['series'][GP]['7']['average'] == 5.0
    assert r['series'][IOS]['7']['average'] == 1.0
    assert r['series']['All']['7']['average'] == 3.0


def test_single_platform_has_no_all_series():
    df = _df([('2024-03-10', 5, '1.0', GP)])
    assert analyzer.get_rating_timeline(df)['platforms'] == [GP]


def test_releases_mark_first_day_of_each_newer_version():
    df = _df([('2024-03-01', 4, '1.0', GP),
              ('2024-03-05', 4, '1.1', GP),
              ('2024-03-06', 4, '1.0', GP),     # eski surumde kalan kullanici: release degil
              ('2024-03-08', 4, '1.10', GP)])   # 1.10 > 1.9 sayisal siralama
    w = analyzer.get_rating_timeline(df, windows=(30,))['series'][GP]['30']

    assert [(r['date'], r['version']) for r in w['releases']] == \
        [('2024-03-05', '1.1'), ('2024-03-08', '1.10')]


def test_releases_are_per_platform_and_labelled_in_all():
    df = _df([('2024-03-01', 4, '9.0', GP), ('2024-03-01', 4, '2.0', IOS),
              ('2024-03-04', 4, '2.1', IOS)])
    r = analyzer.get_rating_timeline(df, windows=(7,))

    assert r['series'][GP]['7']['releases'] == []
    assert r['series']['All']['7']['releases'] == \
        [{'date': '2024-03-04', 'version': '2.1', 'platform': IOS}]


def test_releases_outside_window_are_dropped():
    df = _df([('2024-01-01', 4, '1.0', GP), ('2024-01-02', 4, '1.1', GP),
              ('2024-03-01', 4, '1.1', GP)])
    r = analyzer.get_rating_timeline(df, windows=(7, 90))
    assert r['series'][GP]['7']['releases'] == []
    assert len(r['series'][GP]['90']['releases']) == 1


def test_works_without_version_or_platform_columns():
    df = pd.DataFrame({'score': [4, 2], 'date': pd.to_datetime(['2024-03-01', '2024-03-02'])})
    r = analyzer.get_rating_timeline(df, windows=(7,))
    assert r['available'] is True
    assert r['series'][r['platforms'][0]]['7']['average'] == 3.0


@pytest.mark.parametrize('df', [None, pd.DataFrame(), pd.DataFrame({'score': [4]})])
def test_unavailable_without_dated_scores(df):
    assert analyzer.get_rating_timeline(df)['available'] is False


def test_results_page_renders_average_rating_card(client, patched_fetch):
    resp = client.post('/analyze', data={
        'google_id': 'com.mock.app', 'apple_name': 'mockapp',
        'country': 'us', 'language': 'en', 'max_reviews': '50',
        'complaint_threshold': '2', 'top_words': '10',
    })
    body = resp.data.decode()

    assert resp.status_code == 200
    assert 'Average Rating' in body
    assert 'id="ratingTimelineChart"' in body
    assert 'data-rt-window="30"' in body

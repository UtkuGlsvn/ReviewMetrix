from flask import Blueprint, request, render_template
from . import analyzer

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return render_template('index.html')


def _parse_common_params():
    """Ortak form parametrelerini okuyup doğrular."""
    return {
        'country': request.form['country'],
        'language': request.form['language'],
        'max_reviews': int(request.form['max_reviews']),
        'complaint_threshold': int(request.form['complaint_threshold']),
        'top_words': int(request.form['top_words']),
        'extra_stopwords_str': request.form.get('extra_stopwords', ''),
        'start_date': request.form.get('start_date', '').strip() or None,
        'end_date': request.form.get('end_date', '').strip() or None,
    }


@main_bp.route('/analyze', methods=['POST'])
def analyze_reviews():
    try:
        params = _parse_common_params()
        report = analyzer.build_app_report(
            request.form['google_id'],
            request.form['apple_name'],
            params['country'], params['language'], params['max_reviews'],
            params['complaint_threshold'], params['top_words'],
            params['extra_stopwords_str'],
            params['start_date'], params['end_date'],
        )

        # build_app_report kısmi hatalarda bile summary/dağılımı döndürür
        return render_template(
            'results.html',
            error=report['error'],
            date_range={'start': params['start_date'], 'end': params['end_date']},
            total_reviews=report['total_reviews'],
            complaint_count=report['complaint_count'],
            summary_stats=report['summary_stats'],
            rating_distribution=report['rating_distribution'],
            platform_comparison=report['platform_comparison'],
            most_common_words=report['most_common_words'],
            image_data=report['image_data'],
            sentiment_summary=report['sentiment_summary'],
            trend_data=report['trend_data'],
            sample_reviews=report['sample_reviews'],
            themes=report['themes'],
            emerging_keywords=report['emerging_keywords'],
            version_breakdown=report['version_breakdown'],
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('results.html',
                               error=f"An unexpected error occurred: {e}",
                               summary_stats=None)


@main_bp.route('/compare', methods=['POST'])
def compare_apps():
    """İki uygulamayı yan yana analiz eder."""
    try:
        params = _parse_common_params()

        app_a = analyzer.build_app_report(
            request.form.get('google_id', ''), request.form.get('apple_name', ''),
            params['country'], params['language'], params['max_reviews'],
            params['complaint_threshold'], params['top_words'],
            params['extra_stopwords_str'],
            params['start_date'], params['end_date'],
        )
        app_b = analyzer.build_app_report(
            request.form.get('google_id_b', ''), request.form.get('apple_name_b', ''),
            params['country'], params['language'], params['max_reviews'],
            params['complaint_threshold'], params['top_words'],
            params['extra_stopwords_str'],
            params['start_date'], params['end_date'],
        )

        return render_template('compare.html', app_a=app_a, app_b=app_b)
    except Exception as e:
        print(f"A compare error occurred: {e}")
        return render_template('compare.html',
                               error=f"An unexpected error occurred: {e}")


def _avg_store_rating(summary_stats):
    """summary_stats içindeki Google + iOS puanlarının ortalamasını döndürür (yoksa None)."""
    vals = []
    if summary_stats:
        if summary_stats.get('google') and summary_stats['google'].get('rating'):
            vals.append(summary_stats['google']['rating'])
        if summary_stats.get('ios') and summary_stats['ios'].get('rating'):
            vals.append(summary_stats['ios']['rating'])
    return round(sum(vals) / len(vals), 2) if vals else None


@main_bp.route('/compare-countries', methods=['POST'])
def compare_countries():
    """Aynı uygulamayı birden fazla ülkede analiz edip yan yana karşılaştırır."""
    try:
        params = _parse_common_params()
        google_id = request.form.get('google_id', '')
        apple_name = request.form.get('apple_name', '')

        countries_raw = request.form.get('countries', '')
        # Ülke kodlarını ayıkla, tekrarları kaldır, en fazla 6 ülke
        seen, countries = set(), []
        for c in countries_raw.split(','):
            code = c.strip().lower()
            if code and code not in seen:
                seen.add(code)
                countries.append(code)
        countries = countries[:6]

        if not countries:
            return render_template('compare_countries.html',
                                   error="Please provide at least one country code (e.g. us, gb, de).")

        reports = []
        for code in countries:
            rep = analyzer.build_app_report(
                google_id, apple_name, code, params['language'], params['max_reviews'],
                params['complaint_threshold'], params['top_words'],
                params['extra_stopwords_str'], params['start_date'], params['end_date'],
            )
            rep['country'] = code.upper()
            rep['avg_store_rating'] = _avg_store_rating(rep['summary_stats'])
            rep['avg_complaint_sentiment'] = (
                rep['sentiment_summary']['average_polarity'] if rep['sentiment_summary'] else None
            )
            rep['top_theme'] = rep['themes'][0]['theme'] if rep['themes'] else None
            reports.append(rep)

        chart_data = [{
            'country': r['country'],
            'store_rating': r['avg_store_rating'] or 0,
            'complaints': r['complaint_count'],
            'sentiment': r['avg_complaint_sentiment'] or 0,
            'total': r['total_reviews'],
        } for r in reports]

        return render_template('compare_countries.html',
                               reports=reports, chart_data=chart_data,
                               app_label=(apple_name or google_id or 'App'))
    except Exception as e:
        print(f"A country-compare error occurred: {e}")
        return render_template('compare_countries.html',
                               error=f"An unexpected error occurred: {e}")

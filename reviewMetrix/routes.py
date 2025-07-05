from flask import Blueprint, request, render_template
from . import analyzer

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/analyze', methods=['POST'])
def analyze_reviews():
    summary_stats = None  # Initialize to None
    try:
        # Get data from the form
        google_id = request.form['google_id']
        apple_name = request.form['apple_name']
        country = request.form['country']
        language = request.form['language']
        max_reviews = int(request.form['max_reviews'])
        complaint_threshold = int(request.form['complaint_threshold'])
        top_words = int(request.form['top_words'])
        extra_stopwords_str = request.form.get('extra_stopwords', '')

        all_reviews, summary_stats = analyzer.fetch_reviews_store(
            google_id, apple_name, country, language, max_reviews
        )
        if all_reviews.empty:
            return render_template('results.html', 
                                   summary_stats=summary_stats,
                                   error="No reviews could be found or fetched for the specified apps.")

        complaints = analyzer.preprocess_and_filter_complaints(
            all_reviews, complaint_threshold, apple_name, language, extra_stopwords_str
        )
        
        most_common, image_data, sentiment_summary = None, None, None
        if complaints is not None and not complaints.empty:
            most_common, image_data, sentiment_summary = analyzer.analyze_and_visualize(complaints, top_words)
        else:
            return render_template('results.html', 
                                   summary_stats=summary_stats,
                                   total_reviews=len(all_reviews),
                                   error="No complaint reviews were found in the specified score range.")

        return render_template(
            'results.html',
            total_reviews=len(all_reviews),
            complaint_count=len(complaints),
            most_common_words=most_common,
            image_data=image_data,
            sentiment_summary=sentiment_summary,
            summary_stats=summary_stats
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('results.html', 
                               error=f"An unexpected error occurred: {e}",
                               summary_stats=summary_stats)
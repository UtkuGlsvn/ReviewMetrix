from flask import Blueprint, request, render_template
from . import analyzer

# Create a Blueprint object. 
# 'main' is the name of the blueprint.
main_bp = Blueprint(
    'main', __name__,
    template_folder='templates'
)

@main_bp.route('/')
def index():
    """Renders the main analysis form."""
    return render_template('index.html')

@main_bp.route('/analyze', methods=['POST'])
def analyze_reviews():
    """Handles the form submission, runs the analysis, and renders the results."""
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

        # Run analysis steps using functions from analyzer.py
        all_reviews = analyzer.fetch_reviews_store(google_id, apple_name, country, language, max_reviews)
        if all_reviews.empty:
            return render_template('results.html', error="No reviews could be found or fetched for the specified apps.")

        complaints = analyzer.preprocess_and_filter_complaints(all_reviews, complaint_threshold, apple_name, language, extra_stopwords_str)
        if complaints is None or complaints.empty:
            return render_template('results.html', error="No complaint reviews were found in the specified score range.")

        most_common, image_data = analyzer.analyze_and_visualize(complaints, top_words)
        if not most_common:
            return render_template('results.html', error="No words found to analyze. Reviews might be too short or consist of stop words.")

        return render_template(
            'results.html',
            total_reviews=len(all_reviews),
            complaint_count=len(complaints),
            most_common_words=most_common,
            image_data=image_data
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('results.html', error=f"An unexpected error occurred: {e}")

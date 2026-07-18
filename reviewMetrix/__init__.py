# review_analyzer/__init__.py
from flask import Flask
import nltk

def create_app():
    """Flask application factory."""
    # Create the Flask app instance. This configuration correctly tells Flask
    # to look for a 'templates' folder in this same directory.
    app = Flask(__name__)

    # Download NLTK data if necessary
    download_nltk_data()

    with app.app_context():
        # Import and register the blueprint from routes.py
        from .routes import main_bp
        app.register_blueprint(main_bp)

    return app

def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    try:
        # A more specific check for the 'stopwords' corpus
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK 'stopwords' data not found. Downloading...")
        _download_with_ssl_fallback()
        print("Download complete.")

    # ASO keyword extraction filters to nouns via POS tagging (English only).
    # This is optional: if the tagger is unavailable the analyzer falls back to
    # plain stopword filtering, so failures here must not block startup.
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("NLTK POS tagger not found. Downloading (used for ASO keywords)...")
        try:
            _download_with_ssl_fallback('averaged_perceptron_tagger_eng')
        except Exception as e:
            print(f"POS tagger download failed; ASO keywords will use basic filtering. ({e})")


def _download_with_ssl_fallback(package='stopwords'):
    """Downloads an NLTK package, only relaxing SSL verification for this call
    if the default (verified) attempt fails — instead of disabling it globally."""
    import ssl

    try:
        if nltk.download(package):
            return
    except Exception as e:
        print(f"Verified NLTK download failed, retrying without SSL verification: {e}")

    original_context = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download(package)
    finally:
        ssl._create_default_https_context = original_context

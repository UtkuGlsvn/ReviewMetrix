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
        nltk.download('stopwords')
        print("Download complete.")

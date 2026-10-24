# review_analyzer/__init__.py
from urllib.parse import urlsplit

from flask import Flask, abort, request
import nltk

# Inline <script>/<style> blocks are used throughout the templates, hence
# 'unsafe-inline'; the policy still pins every external origin. Store icons and
# screenshots come from several Google/Apple CDNs, so images allow any https.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data: blob: https:",
    "connect-src 'self'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    'Content-Security-Policy': CONTENT_SECURITY_POLICY,
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
}

def create_app():
    """Flask application factory."""
    # Create the Flask app instance. This configuration correctly tells Flask
    # to look for a 'templates' folder in this same directory.
    app = Flask(__name__)

    # Download NLTK data if necessary
    download_nltk_data()

    @app.before_request
    def reject_cross_site_posts():
        """Lightweight CSRF guard: the app has no sessions, but a foreign page
        could still auto-submit forms to burn a visitor's scrape quota. Browsers
        send Origin (and Sec-Fetch-Site) on POSTs; requests without them, such
        as curl or the test client, are not browser-driven and pass."""
        if request.method != 'POST':
            return None
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            abort(403)
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            abort(403)
        return None

    @app.after_request
    def set_security_headers(response):
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

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
        _download()
        print("Download complete.")

    # ASO keyword extraction filters to nouns via POS tagging (English only).
    # This is optional: if the tagger is unavailable the analyzer falls back to
    # plain stopword filtering, so failures here must not block startup.
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("NLTK POS tagger not found. Downloading (used for ASO keywords)...")
        try:
            _download('averaged_perceptron_tagger_eng')
        except Exception as e:
            print(f"POS tagger download failed; ASO keywords will use basic filtering. ({e})")


def _download(package='stopwords'):
    """Downloads an NLTK package with normal SSL verification.

    There is deliberately no unverified retry: on a broken certificate store the
    fix is to repair it (e.g. run "Install Certificates.command" on macOS), not
    to accept data from whoever answers the connection.
    """
    if not nltk.download(package):
        raise RuntimeError(f"NLTK download of '{package}' failed")

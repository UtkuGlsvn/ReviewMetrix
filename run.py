import logging
import os

from reviewMetrix import create_app

# Send the app's log records (scrape failures, tracebacks) to stderr, where
# gunicorn and Render pick them up
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

app = create_app()

if __name__ == '__main__':
    # Debug is off by default; enable locally with FLASK_DEBUG=1.
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 4999))
    app.run(debug=debug, port=port)

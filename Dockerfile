FROM python:3.12-slim

# wordcloud/matplotlib need a C toolchain to build their wheels on slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # matplotlib and NLTK need writable, predictable locations
    MPLCONFIGDIR=/tmp/matplotlib \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY reviewMetrix/ ./reviewMetrix/
COPY run.py .

# Fetch NLTK corpora at build time so the container never reaches out on first
# request. This calls the application's own downloader rather than duplicating
# the logic, so the SSL fallback and the "tagger is optional" behaviour stay in
# one place. Importing the package does not start the app.
RUN mkdir -p "$NLTK_DATA" \
    && python -c "from reviewMetrix import download_nltk_data; download_nltk_data()" \
    && python -c "import nltk; nltk.data.find('corpora/stopwords')"

# Drop privileges
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Scraping is I/O-bound and slow, so threads are worth more than processes here.
# The timeout is generous on purpose: a 6-country comparison performs six
# sequential scrapes and would be killed by gunicorn's 30s default.
#
# Note: the review cache lives in process memory, so each worker keeps its own.
# More workers means more cache misses; a shared store would be needed to fix
# that properly.
CMD ["gunicorn", "run:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "180", \
     "--access-logfile", "-"]

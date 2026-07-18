# ReviewMetrix — App Review Intelligence

<p align="center">
  <img src="https://raw.githubusercontent.com/UtkuGlsvn/ReviewMetrix/main/img/ReviewMetrix.png" width="350" alt="ReviewMetrix Main Page">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/UtkuGlsvn/ReviewMetrix/main/img/ReviewMetrixAnalys.png" width="350" alt="ReviewMetrix Analysis Page">
</p>

ReviewMetrix is a Flask web app that scrapes user reviews for any app from the
Google Play Store and the Apple App Store, then turns them into an
app-intelligence dashboard: complaint themes, rating distribution, trends,
version regressions, competitor and multi-market comparison, and ASO keyword
gaps.

---

## Features

### Analysis

- **Dual-platform scraping** — Google Play and App Store in one run, merged into a single dataset with a `platform` column.
- **Complaint filtering** — analyze only reviews at or below a score threshold (e.g. ≤ 2 stars).
- **Rating distribution** — 1–5 star breakdown, overall and per platform.
- **Complaint themes** — reviews are auto-categorized into 10 buckets (Crashes & Bugs, Performance, Ads, Login & Account, Price & Payment, UI & UX, Updates, Notifications, Customer Support, Privacy).
- **Sentiment analysis** — polarity per review, aggregated into positive/neutral/negative and plotted over time.
- **Trending complaints** — splits complaints chronologically and surfaces keywords rising in the newer half, flagging ones that are entirely new. Useful for catching a regression early.
- **Version breakdown** — complaints grouped by app version with average rating and sentiment, to spot which release caused a spike (Google Play only; the App Store scraper does not expose review versions).
- **Word cloud and top keywords** — classic frequency view of complaint vocabulary.

### Comparison

- **Competitor comparison** (`/compare`) — two apps side by side: ratings, complaint share, sentiment, theme distribution (as % of complaints), rating distribution and keyword lists.
- **Multi-country comparison** (`/compare-countries`) — the same app across up to 6 markets, with per-country cards and charts for rating, complaint volume and sentiment.

### ASO (App Store Optimization)

- **Listing health** — title and description length against the 30 / 4000 character limits, plus rating and review-volume thresholds, colour-coded.
- **Keyword gap** — words and phrases users repeatedly use in reviews that your listing never mentions. This is the actionable output: running it on Spotify surfaces `apple music` and `youtube music`, i.e. users comparing against competitors.
- **Already targeted** — the top terms your title and description currently cover.

ASO keyword extraction filters to nouns via POS tagging, because ASO keywords
are overwhelmingly nouns rather than verbs or adjectives. Bigrams are kept from
the unfiltered token stream so phrases like `sound quality` survive, but must
contain at least one noun. The tagger is English-only; other languages fall back
to stopword filtering.

> **Scope note:** the keyword gap reflects *user vocabulary*, not search volume.
> Real search-volume data is not exposed by the public store APIs.

### Multilingual support

English, Turkish, German, Spanish and French are supported for both sentiment
and theme detection. Local-language theme keywords are merged with the English
list, since reviews often mix in English terms.

For unsupported languages the app does **not** display a misleading neutral
0.00. Sentiment sections are hidden and an explicit notice is shown; ratings,
themes, keywords and trends remain available.

### Performance and export

- **TTL cache** — scraping results are cached in memory for one hour, keyed by app, country, language and review count (32-entry LRU). Measured on live data: a single app goes 2.42s → 0.22s, a 3-country comparison 6.25s → 0.53s. Empty results are never cached, so a transient scraper failure is not pinned. A **Force refresh** checkbox bypasses the cache.
- **CSV export** of the filtered complaints.
- **PDF export** — a print stylesheet on every report page (white background, chrome hidden, page-break handling); use *Save as PDF*.
- **Quick presets** for nine popular apps, plus an optional date-range filter.

---

## Project Structure

```
ReviewMetrix/
├── reviewMetrix/
│   ├── __init__.py                  # Application factory, NLTK bootstrap
│   ├── routes.py                    # /, /analyze, /compare, /compare-countries
│   ├── analyzer.py                  # Scraping, caching, analysis, ASO
│   └── templates/
│       ├── index.html               # Input form
│       ├── results.html             # Single-app dashboard
│       ├── compare.html             # Two-app comparison
│       └── compare_countries.html   # Multi-market comparison
├── tests/                           # pytest suite (offline)
├── run.py                           # Entry point
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

---

## Setup

### 1. Prerequisites

Python 3.8+, `pip` and `venv`.

### 2. Clone

```bash
git clone https://github.com/UtkuGlsvn/ReviewMetrix.git
cd ReviewMetrix
```

### 3. Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
```

### 4. Install

```bash
pip install -r requirements.txt
```

NLTK stopwords and the POS tagger are downloaded automatically on first run.
The tagger is optional — if the download fails, ASO keyword extraction falls
back to stopword filtering and startup continues.

### 5. Run

```bash
python run.py
```

Available at <http://127.0.0.1:4999>.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_DEBUG` | `0` | Set to `1` for the interactive debugger. Leave off outside local development. |
| `PORT` | `4999` | Port to bind. |

---

## Usage

Fill in the form and choose one of three actions:

| Button | Route | What it does |
| --- | --- | --- |
| **Fetch & Analyze Reviews** | `/analyze` | Full dashboard for one app |
| **Compare Two Apps** | `/compare` | Head-to-head against a competitor |
| **Compare Countries** | `/compare-countries` | The same app across several markets |

Key fields:

- **Google Play App ID** — from the Play Store URL (`id=...`), e.g. `com.instagram.android`
- **Apple Store App Name** — from the App Store URL, e.g. `instagram`
- **Store Country / Review Language** — two-letter codes (`us`, `en`)
- **Complaint Score Threshold** — reviews at or below this rating are treated as complaints
- **From / To Date** — optional date-range filter
- **Force refresh** — skip the cache and re-scrape

---

## Running the Tests

The suite runs fully offline: the scraping layer is patched out, so no network
access is required.

```bash
pip install -r requirements-dev.txt
pytest
```

147 tests covering:

- `tests/test_analyzer.py` — rating distribution, theme categorization, platform comparison, date-range filtering, trending keywords, version breakdown, sentiment, preprocessing, and the `build_app_report` pipeline including every error branch.
- `tests/test_routes.py` — all four routes, error paths, the date-filter badge, country de-duplication and the 6-country cap, and partial failures where one app or market returns no data.
- `tests/test_cache.py` — cache hit/miss, TTL expiry, force refresh, empty results not cached, mutation isolation, LRU eviction order, and route-level caching.
- `tests/test_multilingual.py` — per-language sentiment and themes, negation handling, and the UI degradation for unsupported languages.
- `tests/test_aso.py` — listing health thresholds, keyword gap, phrase detection, noise filtering, and graceful fallback without the POS tagger.

Test data is deliberately discriminating — ordering fixtures use unequal counts
so sort direction is actually asserted, rather than passing vacuously.

---

## Technologies

| Layer | Stack |
| --- | --- |
| Backend | Python, Flask (application factory + blueprints) |
| Data | pandas, NLTK, TextBlob |
| Charts | ApexCharts, Matplotlib, WordCloud |
| Scraping | `google-play-scraper`, `app-store-scraper`, iTunes Search API |
| Testing | pytest |

---

## Known Limitations

- **App Store scraping is fragile.** `app-store-scraper` breaks when Apple changes its endpoints, and it does not expose the app version per review, so the version breakdown is Google Play only. Review fetching failures are caught, so the rest of the report still renders.
- **Sentiment outside English is lexicon-based.** It handles clearly polar wording and negation, but factual complaints with no polar words score 0.00. It is not a transformer model, and accuracy expectations should be set accordingly.
- **The cache is per-process.** Running multiple workers gives each its own cache; a shared store (e.g. Redis) would be needed for a real deployment.
- **The dev server is not production-ready.** Use a WSGI server such as gunicorn behind a reverse proxy.

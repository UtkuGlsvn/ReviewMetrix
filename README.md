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

The report is split into four tabs — **Overview**, **Issues**, **ASO** and
**Trends & Reviews** — so each view stays readable instead of forming one very
long scroll. Issues and ASO show a count badge, and PDF export covers the open
tab only.

### Analysis

- **Dual-platform scraping** — Google Play and App Store in one run, merged into a single dataset with a `platform` column.
- **Complaint filtering** — analyze only reviews at or below a score threshold (e.g. ≤ 2 stars).
- **Rating momentum** — the store only ever shows the lifetime average, where millions of old reviews can hide a present-day collapse. This compares it against the scraped window and labels the gap. On Spotify: 4.34 lifetime vs 3.81 across the last 200 reviews.
- **Fix First prioritization** — themes ranked by `(complaints + likes) × severity` rather than raw count, where severity is `5 − average rating`. A theme with two complaints, eight likes and a 1.0 average outranks one with seven mild complaints.
- **Rating distribution** — 1–5 star breakdown, overall and per platform.
- **Complaint themes** — reviews are auto-categorized into 10 buckets (Crashes & Bugs, Performance, Ads, Login & Account, Price & Payment, UI & UX, Updates, Notifications, Customer Support, Privacy).
- **Sentiment analysis** — polarity per review, aggregated into positive/neutral/negative and plotted over time.
- **Trending complaints** — splits complaints chronologically and surfaces keywords rising in the newer half, flagging ones that are entirely new. Useful for catching a regression early.
- **Developer responses** — reply rate overall and specifically on complaints, median reply time, and a per-theme breakdown sorted worst-first so ignored themes surface. The complaint-specific rate is the meaningful one: an app can look responsive while only answering its praise.
- **Version breakdown** — complaints grouped by app version with average rating and sentiment, to spot which release caused a spike.
- **Word cloud and top keywords** — classic frequency view of complaint vocabulary.

Review likes (`thumbsUpCount`) are treated as additional affected users: they
feed the priority score and sort the review table, so the complaint the most
people endorsed appears first.

### Comparison

- **Competitor comparison** (`/compare`) — two apps side by side: ratings, complaint share, sentiment, theme distribution (as % of complaints), rating distribution and keyword lists.
- **Competitor ASO gap** — splits listing vocabulary into "they target, you don't" and "you target, they don't". Brand and developer names are excluded; they are not transferable keywords.
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
- **CSV export** of the filtered complaints, including likes.
- **PDF export** — scoped to the open tab, via a print stylesheet (white background, chrome hidden, page-break handling) plus a header carrying the section name, app title and review counts. Charts render lazily per tab, so a tab must be opened before it can be printed.
- **Quick presets** for nine popular apps, plus an optional date-range filter.

---

## Project Structure

```
ReviewMetrix/
├── reviewMetrix/
│   ├── __init__.py                  # Application factory, NLTK bootstrap
│   ├── routes.py                    # /, /analyze, /compare, /compare-countries
│   ├── analyzer.py                  # Scraping, caching, analysis, ASO
│   ├── ratelimit.py                 # Per-IP scrape budget
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
| `RATE_LIMIT_MAX` | `30` | Scrape units allowed per IP per window. |
| `RATE_LIMIT_WINDOW` | `3600` | Rate-limit window in seconds. |
| `TRUST_PROXY` | unset | Set to `1` only when behind a proxy that rewrites `X-Forwarded-For`. |
| `CACHE_TTL_SECONDS` | `3600` | How long scraped reviews stay cached. |
| `CACHE_MAX_ENTRIES` | `32` | Cached scrapes kept in memory. The main memory dial. |

### Rate limiting

The scraping routes are limited per IP, because the stores rate-limit and
eventually block callers. The limiter counts **scrape units, not requests**: a
single analysis costs 1, a competitor comparison costs 2, and a six-country
comparison costs 6. Over-budget requests are rejected whole with a `429` and a
`Retry-After` header, and rejected attempts are not recorded, so retrying while
blocked cannot push your own reset further away.

`X-Forwarded-For` is ignored unless `TRUST_PROXY=1`, since any client can set
that header and would otherwise mint a fresh IP per request.

The counters live in process memory. With more than one worker each gets its
own, so the effective limit multiplies by the worker count — see the note on
workers below.

`run.py` uses the Werkzeug development server. For anything beyond local use,
serve the same `run:app` entrypoint with gunicorn:

```bash
gunicorn run:app --bind 0.0.0.0:8000 --workers 1 --threads 8 --timeout 180
```

One worker with threads, not several processes. Scraping is I/O-bound so
threads carry the concurrency, and both the review cache and the rate-limit
counters live in process memory — every extra worker gets its own copy, which
means more cache misses and an effective rate limit multiplied by the worker
count. The long timeout is deliberate: a six-country comparison performs six
sequential scrapes and would be killed by gunicorn's 30-second default.

### Hosting notes

This is a server-side app: it runs Python on every request. **GitHub Pages
cannot host it** — Pages serves static files only. Anything that runs a Python
process works (Render, Fly.io, Railway, Koyeb, PythonAnywhere, Hugging Face
Spaces).

Measured memory on this app: **~126 MB idle, ~326 MB with a warm cache.** A
512 MB free tier fits one worker; lower `CACHE_MAX_ENTRIES` if you need more
headroom.

One risk worth knowing before you deploy publicly: the stores block datacenter
IP ranges far more readily than residential ones, so scraping that works from a
laptop can return 403s or empty results from a cloud host. Rate limiting
reduces the pressure but does not remove that risk.

### Docker

```bash
docker build -t reviewmetrix .
docker run -p 8000:8000 reviewmetrix
```

The image pre-downloads the NLTK corpora at build time by calling the app's own
downloader, so the container does not reach out on its first request, and it
runs as a non-root user.

> The review cache lives in process memory, so each gunicorn worker keeps its
> own copy. More workers means more cache misses. A shared store such as Redis
> would be needed to fix that properly.

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

222 tests covering:

- `tests/test_analyzer.py` — rating distribution, theme categorization, platform comparison, date-range filtering, trending keywords, version breakdown, sentiment, preprocessing, and the `build_app_report` pipeline including every error branch.
- `tests/test_routes.py` — all four routes, error paths, the date-filter badge, country de-duplication and the 6-country cap, and partial failures where one app or market returns no data.
- `tests/test_cache.py` — cache hit/miss, TTL expiry, force refresh, empty results not cached, mutation isolation, LRU eviction order, and route-level caching.
- `tests/test_multilingual.py` — per-language sentiment and themes, negation handling, and the UI degradation for unsupported languages.
- `tests/test_aso.py` — listing health thresholds, keyword gap, phrase detection, noise filtering, and graceful fallback without the POS tagger.
- `tests/test_priorities.py` — momentum thresholds and per-platform scoping, priority ranking against raw volume, likes as affected users, and review ordering.
- `tests/test_responses.py` — reply rates, median reply time, per-theme gaps, competitor keyword gap, and the POS context fix.
- `tests/test_ratelimit.py` — window mechanics, per-unit costs, rejected requests not being recorded, IP isolation, stale-key eviction and the X-Forwarded-For trust boundary.

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
| Serving | gunicorn, Docker |

---

## Known Limitations

- **Several signals are Google Play only.** The App Store scraper exposes neither the app version per review, nor review likes, nor developer replies — so the version breakdown, likes weighting and response analysis cover Google Play alone, and the UI labels them as such. Ratings, themes, sentiment and keywords use both stores.
- **App Store scraping is fragile.** `app-store-scraper` breaks when Apple changes its endpoints. Review fetching failures are caught, so the rest of the report still renders.
- **Sentiment outside English is lexicon-based.** It handles clearly polar wording and negation, but factual complaints with no polar words score 0.00. It is not a transformer model, and accuracy expectations should be set accordingly.
- **The cache is per-process.** Running multiple workers gives each its own cache; a shared store (e.g. Redis) would be needed for a real deployment.
- **The dev server is not production-ready.** Use a WSGI server such as gunicorn behind a reverse proxy.

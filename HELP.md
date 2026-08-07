# ReviewMetrix — Help & User Guide

A practical guide to every field, tab and metric. For setup and deployment see
the [README](README.md).

---

## Getting started

1. Open the app (default `http://127.0.0.1:4999`).
2. Enter an app, or pick one from **Quick Preset**.
3. Click **Fetch & Analyze Reviews**.

You can paste a **full store URL** into either field — it is parsed to the id
or slug automatically:

- Google Play: `https://play.google.com/store/apps/details?id=com.spotify.music` → `com.spotify.music`
- App Store: `https://apps.apple.com/us/app/spotify-music-and-podcasts/id324684580` → `spotify-music-and-podcasts`

---

## The form, field by field

| Field | What it does | Notes |
| --- | --- | --- |
| **Quick Preset** | Auto-fills a popular app's id and name | Just a shortcut |
| **Google Play App ID** | The package id (`com.instagram.android`) | Or paste the Play Store URL |
| **Apple Store App Name** | The name slug (`instagram`) | Or paste the App Store URL. This is the slug, **not** the numeric id |
| **Store Country** | Two-letter market code (`us`, `gb`, `de`) | Affects which reviews are fetched |
| **Review Language** | Two-letter language (`en`, `tr`, `de`, `es`, `fr`) | **Also drives sentiment and theme detection** — see Multilingual below |
| **Number of Reviews** | How many recent reviews to pull | More = steadier stats, slower. Capped at 1000 |
| **Complaint Threshold (≤)** | Ratings at or below this count as complaints | Usually 2 |
| **Top Words** | How many keywords to show | |
| **From / To Date** | Restrict analysis to a date range | Optional |
| **Extra Stopwords** | Words to ignore in the analysis | e.g. for a game: `game, level, play` |
| **Force refresh** | Skip the 1-hour cache and re-scrape | Slower, guarantees the latest |

### The three buttons

- **Fetch & Analyze Reviews** → single-app dashboard
- **Compare Two Apps** → head-to-head; fill the *Compare with a competitor* section first
- **Compare Countries** → same app across markets; fill the *Compare across countries* section first

If you click a comparison button without filling its section, the app opens the
section and highlights the missing field instead of running an empty comparison.

---

## Reading the results

The dashboard is split into four tabs. Every section heading has a **?** icon —
hover (or tap on mobile) for a one-line explanation.

### Overview tab — "What's the state?"

- **Rating Momentum** — the store always shows the *lifetime* average, which
  millions of old reviews can prop up. This compares it against the reviews you
  just scraped. A red **Declining** badge means recent reviews rate the app
  lower than its lifetime score — the single most important number here.
- **Platform Comparison** — average rating and sentiment on each store, so you
  can tell whether a problem is Android- or iOS-specific.
- **Rating Distribution** — the 1–5 star spread. An average of 3.5 built from
  many 1s and 5s is a different story from one built from steady 3s and 4s.

### Issues tab — "What do I fix?"

- **Complaint Themes** — complaints auto-grouped into 10 categories.
- **Fix First** — themes ranked by impact, not raw count. The score is
  `(complaints + likes) × severity`, where severity is `5 − average rating`.
  A theme with a few harsh, heavily-liked complaints outranks one with many
  mild ones.
- **Trending Complaints** — words rising in the newer half of the reviews. A
  **NEW** badge means it was absent before, which often signals a regression
  from a recent release.
- **Developer Responses** — reply rate (overall and on complaints), median
  reply time, and which themes go unanswered. *Google Play only.*

### ASO tab — "How do I get found?"

- **Listing Health** — title and description length against the store limits,
  plus rating, review volume and screenshot count, colour-coded.
- **Keyword Opportunities** — words and phrases users say in reviews that your
  listing never mentions. The actionable output.
- **Already Targeted** — what your title and description currently cover.
- **Screenshots** — the store gallery, pulled from the listing.

### Trends & Reviews tab — the detail

Version breakdown (which release drew complaints), the rating/sentiment trend
over time, top keywords, the word cloud, and the raw review table sorted with
the most-liked complaints first.

---

## Comparison pages

- **Two apps** — a versus header, head-to-head metrics, theme and rating
  distributions as percentages (fair across different-sized apps), an **ASO
  Listing Strength** table, and an **ASO Listing Gap** (what each app targets
  that the other doesn't).
- **Countries** — per-market cards and charts for rating, complaint volume and
  sentiment. Answers "is this problem global, or specific to one market?"

---

## Multilingual behaviour

Sentiment and themes are supported in **English, Turkish, German, Spanish and
French**. The **Review Language** field selects which.

For any other language the app does **not** show a misleading neutral 0.00 —
sentiment sections are hidden with a notice, while ratings, themes, keywords and
trends stay available.

---

## Exports

- **CSV** — the filtered complaints, including likes.
- **PDF** — *Save as PDF* prints the **currently open tab only**. Open a tab
  before printing it, since its charts render when the tab is first shown.

---

## Content prompts

The **📝 SEO / ASO / AIO content prompts** link (top of the form and in the
footer) opens a library of copy-ready prompts:

- **SEO** — keyword clustering, title/meta, content briefs, internal linking, FAQ schema.
- **ASO** — store title/subtitle, the iOS keyword field, long descriptions, review replies, screenshot copy.
- **AIO** — AI Optimization / Generative Engine Optimization: making content answer-first and citable by AI answer engines.

Each prompt has a **Copy** button and `{placeholder}` fields to fill with your
own details before pasting into an AI assistant.

**Adding your own prompt:** drop a Markdown file into
`reviewMetrix/prompts/{seo,aso,aio}/` with this header, and it appears
automatically:

```markdown
---
title: Your Prompt Title
description: One line describing it.
---
The prompt body, with {placeholders} for the user to fill in.
```

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| **"No reviews could be found"** | Wrong app id / name, or the app has no reviews in that country. Check the id/slug. |
| **"Unknown App" in a comparison** | The second app's id/name is wrong or was left blank. |
| **Sentiment sections missing** | The review language isn't one of the five supported — expected behaviour, not a bug. |
| **Version / likes / replies empty** | These come from Google Play only; the App Store scraper doesn't expose them. |
| **"Rate limit reached" (429)** | The per-IP scrape budget for the hour is used up. Wait, or raise `RATE_LIMIT_MAX`. |
| **Empty or 403 results after deploy** | Stores block datacenter IPs more readily than home ones. See the README hosting notes. |

---

## Known limitations

- App Store scraping is fragile and breaks when Apple changes its endpoints.
- Non-English sentiment is lexicon-based: it catches clearly polar wording but
  scores neutral, factual complaints at 0.00.
- The keyword gap reflects user vocabulary, not real search volume (which the
  stores don't expose).
- The cache and rate limiter are per-process; running multiple workers gives
  each its own.

# reviewMetrix/analyzer.py
import pandas as pd
from nltk.corpus import stopwords
import re
import time
import copy
import threading
from collections import Counter, OrderedDict
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import requests
import urllib.parse

# Import TextBlob for sentiment analysis
from textblob import TextBlob

from google_play_scraper import reviews, Sort, app as google_app_details
from app_store_scraper import AppStore

def fetch_reviews_store(google_id, apple_name, country, lang, max_reviews_to_fetch):
    all_dfs = []
    summary_stats = {'google': None, 'ios': None}

    # --- Google Play ---
    try:
        google_reviews_data, _ = reviews(google_id, lang=lang, country=country, sort=Sort.NEWEST, count=max_reviews_to_fetch)
        if google_reviews_data:
            google_df = pd.DataFrame(google_reviews_data)
            cols = ['content', 'score', 'at']
            rename = {'content': 'review', 'at': 'date'}
            # Google Play exposes the app version a review was written against
            if 'reviewCreatedVersion' in google_df.columns:
                cols.append('reviewCreatedVersion')
                rename['reviewCreatedVersion'] = 'version'
            # Likes show how many other users endorsed the same complaint
            if 'thumbsUpCount' in google_df.columns:
                cols.append('thumbsUpCount')
                rename['thumbsUpCount'] = 'likes'
            # Developer replies are a reputation-management signal
            if 'replyContent' in google_df.columns:
                cols.append('replyContent')
                rename['replyContent'] = 'reply'
            if 'repliedAt' in google_df.columns:
                cols.append('repliedAt')
                rename['repliedAt'] = 'replied_at'
            google_df_renamed = google_df[cols].rename(columns=rename)
            google_df_renamed['platform'] = 'Google Play'
            if 'version' not in google_df_renamed.columns:
                google_df_renamed['version'] = None
            if 'likes' not in google_df_renamed.columns:
                google_df_renamed['likes'] = 0
            for col in ('reply', 'replied_at'):
                if col not in google_df_renamed.columns:
                    google_df_renamed[col] = None
            all_dfs.append(google_df_renamed)
        
        gp_details = google_app_details(google_id, lang=lang, country=country)
        summary_stats['google'] = {
            'rating': float(gp_details.get('score', 0)), # Float'a zorluyoruz
            'reviews': int(gp_details.get('reviews', 0)), # Int'e zorluyoruz
            'title': gp_details.get('title', 'Unknown'),
            'developer': gp_details.get('developer', 'Unknown'),
            'category': gp_details.get('genre', 'Unknown'),
            'installs': gp_details.get('installs', 'Unknown'),
            'version': gp_details.get('version', 'Unknown'),
            'released': gp_details.get('released', 'Unknown'),
            'description': gp_details.get('description', '')[:200] + '...' if gp_details.get('description') else 'No description',
            # ASO needs the full text; the truncated field above is for display only
            'description_full': gp_details.get('description', '') or '',
        }
    except Exception as e:
        print(f"Google Play fetch error->: {e}")

    # --- Apple Store: metadata via the iTunes API, reviews via the scraper ---
    try:
        safe_apple_name = urllib.parse.quote(apple_name)
        itunes_url = f"https://itunes.apple.com/search?term={safe_apple_name}&country={country}&entity=software&limit=1"
        itunes_resp = requests.get(itunes_url)
        
        if itunes_resp.status_code == 200 and itunes_resp.json().get('resultCount', 0) > 0:
            app_data = itunes_resp.json()['results'][0]
            summary_stats['ios'] = {
                'rating': float(app_data.get('averageUserRating', 0)),
                'reviews': int(app_data.get('userRatingCount', 0)),
                'title': app_data.get('trackName', 'Unknown'),
                'developer': app_data.get('sellerName', 'Unknown'),
                'category': app_data.get('primaryGenreName', 'Unknown'),
                'version': app_data.get('version', 'Unknown'),
                'released': app_data.get('releaseDate', 'Unknown')[:10] if app_data.get('releaseDate') else 'Unknown',
                'description': app_data.get('description', '')[:200] + '...' if app_data.get('description') else 'No description',
                'description_full': app_data.get('description', '') or '',
            }
    except Exception as e:
        print(f"iTunes API Error ->: {e}")

    try:
        apple_scraper = AppStore(country=country, app_name=apple_name)
        apple_scraper.review(how_many=max_reviews_to_fetch)
        if apple_scraper.reviews:
            apple_df = pd.DataFrame(apple_scraper.reviews)
            apple_df_renamed = apple_df[['review', 'rating', 'date']].rename(columns={'rating': 'score'})
            apple_df_renamed['platform'] = 'App Store'
            apple_df_renamed['version'] = None  # not exposed by the App Store scraper
            apple_df_renamed['likes'] = 0       # not exposed by the App Store scraper
            apple_df_renamed['reply'] = None    # not exposed by the App Store scraper
            apple_df_renamed['replied_at'] = None
            all_dfs.append(apple_df_renamed)
    except Exception as e:  
        print(f"Apple Store fetch review error ->: {e}")
    
    reviews_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return reviews_df, summary_stats


# ---------------------------------------------------------------------------
# TTL cache for scraping results
#
# Store scraping is slow and rate-limited. When the same (app, country,
# language, review count) is requested again within the window we serve it
# from memory instead of re-scraping. Multi-country comparison benefits most,
# since it performs one scrape per market.
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS = 3600   # 1 hour
CACHE_MAX_ENTRIES = 32     # bounds memory use; oldest entries are evicted

_review_cache = OrderedDict()  # key -> (timestamp, DataFrame, summary_stats)
_cache_lock = threading.Lock()


def clear_review_cache():
    """Empty the cache completely (used by tests and manual refresh)."""
    with _cache_lock:
        _review_cache.clear()


def review_cache_info():
    """Return the current cache state."""
    with _cache_lock:
        return {
            'entries': len(_review_cache),
            'ttl_seconds': CACHE_TTL_SECONDS,
            'max_entries': CACHE_MAX_ENTRIES,
        }


def fetch_reviews_cached(google_id, apple_name, country, lang,
                         max_reviews_to_fetch, ttl=None):
    """TTL-cached wrapper around fetch_reviews_store.

    ttl=0 skips the cache (forced refresh). Empty results are never cached, so
    a transient scraper failure is not pinned for an hour. Returns copies, so
    callers cannot corrupt the cached entry.
    """
    ttl = CACHE_TTL_SECONDS if ttl is None else ttl
    key = (google_id, apple_name, country, lang, max_reviews_to_fetch)
    now = time.time()

    with _cache_lock:
        entry = _review_cache.get(key)
        if entry is not None:
            cached_at, cached_df, cached_stats = entry
            if now - cached_at < ttl:
                _review_cache.move_to_end(key)  # mark as most recently used
                return cached_df.copy(), copy.deepcopy(cached_stats)
            del _review_cache[key]  # expired

    reviews_df, summary_stats = fetch_reviews_store(
        google_id, apple_name, country, lang, max_reviews_to_fetch
    )

    if reviews_df is not None and not reviews_df.empty:
        with _cache_lock:
            _review_cache[key] = (time.time(), reviews_df.copy(),
                                  copy.deepcopy(summary_stats))
            _review_cache.move_to_end(key)
            while len(_review_cache) > CACHE_MAX_ENTRIES:
                _review_cache.popitem(last=False)  # evict least recently used

    return reviews_df, summary_stats


# Keyword -> theme map. A review counts toward a theme if it contains any of
# that theme's trigger words.
COMPLAINT_THEMES = {
    'Crashes & Bugs': ['crash', 'bug', 'freeze', 'froze', 'glitch', 'error', 'broken', 'stuck', 'hang', 'force close', 'not working', "doesn't work", 'wont open', "won't open'"],
    'Performance': ['slow', 'lag', 'laggy', 'battery', 'drain', 'heat', 'memory', 'storage', 'loading', 'speed', 'performance'],
    'Ads': ['ad', 'ads', 'advert', 'advertisement', 'popup', 'pop up', 'spam'],
    'Login & Account': ['login', 'log in', 'sign in', 'signin', 'account', 'password', 'verification', 'otp', 'authenticate', 'locked out', 'ban', 'banned', 'suspended'],
    'Price & Payment': ['price', 'expensive', 'subscription', 'refund', 'charge', 'charged', 'payment', 'money', 'pay', 'premium', 'purchase', 'billing'],
    'UI & UX': ['ui', 'ux', 'design', 'layout', 'interface', 'confusing', 'ugly', 'update ruined', 'new update', 'hard to use', 'complicated', 'cluttered'],
    'Updates': ['update', 'version', 'downgrade', 'rollback', 'latest version', 'after update'],
    'Notifications': ['notification', 'notifications', 'alert', 'spammy', 'ringtone'],
    'Customer Support': ['support', 'customer service', 'help', 'response', 'contact', 'ignored', 'no reply'],
    'Privacy & Ads Tracking': ['privacy', 'data', 'permission', 'tracking', 'security', 'hacked', 'scam'],
}

# The same themes in other languages. For a non-English analysis these are
# merged with the English list, since reviews often mix in English terms.
THEME_KEYWORDS_BY_LANG = {
    'tr': {
        'Crashes & Bugs': ['çöküyor', 'çökme', 'çöktü', 'kapanıyor', 'hata', 'donuyor', 'donma', 'açılmıyor', 'çalışmıyor', 'bozuk', 'kasıyor'],
        'Performance': ['yavaş', 'ağır', 'gecikme', 'pil', 'batarya', 'ısınıyor', 'yükleniyor', 'geç'],
        'Ads': ['reklam', 'reklamlar', 'reklamı'],
        'Login & Account': ['giriş', 'oturum', 'hesap', 'şifre', 'doğrulama', 'banlandı', 'engellendi', 'kapatıldı'],
        'Price & Payment': ['pahalı', 'ücret', 'abonelik', 'para', 'ödeme', 'iade', 'fiyat', 'satın'],
        'UI & UX': ['arayüz', 'tasarım', 'karışık', 'kullanışsız', 'anlaşılmıyor', 'karmaşık'],
        'Updates': ['güncelleme', 'güncellendi', 'güncellemeden', 'sürüm'],
        'Notifications': ['bildirim', 'bildirimler'],
        'Customer Support': ['destek', 'müşteri', 'yanıt', 'iletişim', 'dönüş'],
        'Privacy & Ads Tracking': ['gizlilik', 'veri', 'izin', 'güvenlik', 'dolandırıcı'],
    },
    'de': {
        'Crashes & Bugs': ['absturz', 'stürzt', 'abstürze', 'fehler', 'hängt', 'kaputt', 'defekt', 'friert'],
        'Performance': ['langsam', 'träge', 'akku', 'batterie', 'heiß', 'ladezeit', 'lädt'],
        'Ads': ['werbung', 'anzeigen', 'werbe'],
        'Login & Account': ['anmelden', 'anmeldung', 'einloggen', 'konto', 'passwort', 'gesperrt'],
        'Price & Payment': ['teuer', 'preis', 'abo', 'abonnement', 'geld', 'zahlung', 'erstattung', 'kosten'],
        'UI & UX': ['oberfläche', 'design', 'unübersichtlich', 'kompliziert', 'verwirrend'],
        'Updates': ['update', 'aktualisierung', 'version'],
        'Notifications': ['benachrichtigung', 'benachrichtigungen', 'mitteilungen'],
        'Customer Support': ['support', 'kundendienst', 'antwort', 'kontakt'],
        'Privacy & Ads Tracking': ['datenschutz', 'daten', 'berechtigung', 'sicherheit', 'betrug'],
    },
    'es': {
        'Crashes & Bugs': ['falla', 'fallo', 'cierra', 'error', 'cuelga', 'bloquea', 'no funciona'],
        'Performance': ['lento', 'lenta', 'batería', 'calienta', 'carga', 'tarda'],
        'Ads': ['anuncios', 'publicidad', 'anuncio'],
        'Login & Account': ['iniciar', 'sesión', 'cuenta', 'contraseña', 'bloqueado', 'verificación'],
        'Price & Payment': ['caro', 'precio', 'suscripción', 'dinero', 'pago', 'reembolso', 'cobro'],
        'UI & UX': ['interfaz', 'diseño', 'confuso', 'complicado'],
        'Updates': ['actualización', 'actualizar', 'versión'],
        'Notifications': ['notificación', 'notificaciones'],
        'Customer Support': ['soporte', 'atención', 'respuesta', 'contacto'],
        'Privacy & Ads Tracking': ['privacidad', 'datos', 'permiso', 'seguridad', 'estafa'],
    },
    'fr': {
        'Crashes & Bugs': ['plante', 'plantage', 'bug', 'erreur', 'ferme', 'bloque', 'ne fonctionne pas'],
        'Performance': ['lent', 'lente', 'lenteur', 'batterie', 'chauffe', 'charge'],
        'Ads': ['publicité', 'publicités', 'pubs', 'pub'],
        'Login & Account': ['connexion', 'connecter', 'compte', 'mot de passe', 'bloqué', 'vérification'],
        'Price & Payment': ['cher', 'chère', 'prix', 'abonnement', 'argent', 'paiement', 'remboursement'],
        'UI & UX': ['interface', 'design', 'confus', 'compliqué'],
        'Updates': ['mise à jour', 'version', 'mise'],
        'Notifications': ['notification', 'notifications'],
        'Customer Support': ['support', 'service client', 'réponse', 'contact'],
        'Privacy & Ads Tracking': ['confidentialité', 'données', 'autorisation', 'sécurité', 'arnaque'],
    },
}


def get_theme_keywords(lang_code='en'):
    """Return the theme -> keyword map for a language, merged with English."""
    base = {theme: list(words) for theme, words in COMPLAINT_THEMES.items()}
    extra = THEME_KEYWORDS_BY_LANG.get((lang_code or 'en').lower())
    if extra:
        for theme, words in extra.items():
            base.setdefault(theme, []).extend(words)
    return base


# Gap in stars between lifetime and current rating before it counts as a trend
MOMENTUM_DECLINE_THRESHOLD = -0.3
MOMENTUM_IMPROVE_THRESHOLD = 0.3


def get_momentum(summary_stats, all_reviews):
    """Compare the store's lifetime average against the scraped window.

    The store only ever shows the lifetime average, where millions of old
    reviews can hide a present-day collapse. The gap answers "is this app
    declining?" directly.
    """
    result = {'platforms': [], 'available': False}
    if all_reviews is None or all_reviews.empty or 'score' not in all_reviews.columns:
        return result

    try:
        work = all_reviews.copy()
        work['score'] = pd.to_numeric(work['score'], errors='coerce')
        work = work.dropna(subset=['score'])
        if work.empty:
            return result

        stats = summary_stats or {}
        store_map = {'Google Play': stats.get('google'), 'App Store': stats.get('ios')}

        for platform, store in store_map.items():
            if not store or not store.get('rating'):
                continue
            if 'platform' in work.columns:
                subset = work[work['platform'] == platform]
            else:
                subset = work
            if subset.empty:
                continue

            lifetime = float(store['rating'])
            recent = float(subset['score'].mean())
            delta = recent - lifetime

            if delta <= MOMENTUM_DECLINE_THRESHOLD:
                status = 'declining'
            elif delta >= MOMENTUM_IMPROVE_THRESHOLD:
                status = 'improving'
            else:
                status = 'stable'

            result['platforms'].append({
                'platform': platform,
                'lifetime': round(lifetime, 2),
                'recent': round(recent, 2),
                'delta': round(delta, 2),
                'status': status,
                'sample': int(len(subset)),
            })

        result['available'] = bool(result['platforms'])
    except Exception as e:
        print(f"Error computing momentum: {e}")
    return result


def get_response_analysis(df, lang_code='en'):
    """Analyse the developer's replies to reviews.

    Reply rate and speed are a reputation signal; which themes go unanswered
    is a directly actionable gap. Reply data is Google Play only.
    """
    result = {
        'available': False,
        'total': 0,
        'replied': 0,
        'response_rate': 0.0,
        'complaints_total': 0,
        'complaints_replied': 0,
        'complaint_response_rate': 0.0,
        'median_hours': None,
        'by_theme': [],
    }
    if df is None or df.empty or 'reply' not in df.columns:
        return result

    try:
        work = df.copy()
        # Only score the platform that exposes a reply field
        if 'platform' in work.columns:
            work = work[work['platform'] == 'Google Play']
        if work.empty:
            return result

        has_reply = work['reply'].notna() & (work['reply'].astype(str).str.strip() != '')
        result['total'] = int(len(work))
        result['replied'] = int(has_reply.sum())
        result['response_rate'] = round(result['replied'] * 100 / len(work), 1)

        # The rate on complaints matters more: an app can look responsive
        # while only answering its praise
        if 'score' in work.columns:
            scores = pd.to_numeric(work['score'], errors='coerce')
            complaint_mask = scores <= 2
            complaints = work[complaint_mask]
            if not complaints.empty:
                c_replied = complaints['reply'].notna() & (complaints['reply'].astype(str).str.strip() != '')
                result['complaints_total'] = int(len(complaints))
                result['complaints_replied'] = int(c_replied.sum())
                result['complaint_response_rate'] = round(
                    result['complaints_replied'] * 100 / len(complaints), 1)

        # Reply latency in hours
        if 'replied_at' in work.columns and result['replied']:
            replied_rows = work[has_reply]
            posted = pd.to_datetime(replied_rows['date'], errors='coerce')
            answered = pd.to_datetime(replied_rows['replied_at'], errors='coerce')
            for series in (posted, answered):
                try:
                    series = series.dt.tz_localize(None)
                except (TypeError, AttributeError):
                    pass
            delta_hours = (answered - posted).dt.total_seconds() / 3600
            delta_hours = delta_hours.dropna()
            delta_hours = delta_hours[delta_hours >= 0]
            if not delta_hours.empty:
                result['median_hours'] = round(float(delta_hours.median()), 1)

        # Which themes get answered, and which are ignored?
        theme_keywords = get_theme_keywords(lang_code)
        texts = work['review'].fillna('').astype(str).str.lower()
        by_theme = []
        for theme, keywords in theme_keywords.items():
            match = texts.apply(lambda t: any(kw in t for kw in keywords))
            total = int(match.sum())
            if total == 0:
                continue
            replied = int((match & has_reply).sum())
            by_theme.append({
                'theme': theme,
                'total': total,
                'replied': replied,
                'rate': round(replied * 100 / total, 1),
            })
        by_theme.sort(key=lambda t: (t['rate'], -t['total']))
        result['by_theme'] = by_theme

        result['available'] = result['total'] > 0
    except Exception as e:
        print(f"Error analyzing developer responses: {e}")
    return result


def get_competitor_keyword_gap(stats_a, stats_b, lang_code='en', top_n=12):
    """Compare two apps' store listings.

    - b_only: terms the competitor targets and you do not (your gap)
    - a_only: terms you target and they do not (your differentiation)
    """
    result = {'available': False, 'a_only': [], 'b_only': [], 'shared': 0}

    def listing_counts(stats):
        parts = []
        for key in ('google', 'ios'):
            store = (stats or {}).get(key)
            if store:
                parts.append(store.get('title') or '')
                parts.append(store.get('description_full') or '')
        text = ' '.join(parts)
        if not text.strip():
            return Counter()
        tokens = _aso_tokens(text, lang_code)
        # Connectives like "based", "including", "seamlessly" are not ASO
        # keywords; single-word targets are nouns. Tagging runs on the raw
        # listing text so the tagger keeps its context.
        return Counter(_noun_filter(tokens, lang_code, context_text=text))

    def brand_words(stats):
        """An app's own brand name is not a transferable ASO keyword."""
        words = set()
        for key in ('google', 'ios'):
            store = (stats or {}).get(key)
            if store:
                words |= set(_aso_tokens(store.get('title') or '', lang_code))
                words |= set(_aso_tokens(store.get('developer') or '', lang_code))
        return words

    try:
        counts_a = listing_counts(stats_a)
        counts_b = listing_counts(stats_b)
        if not counts_a and not counts_b:
            return result

        brands = brand_words(stats_a) | brand_words(stats_b)
        for term in brands:
            counts_a.pop(term, None)
            counts_b.pop(term, None)

        set_a, set_b = set(counts_a), set(counts_b)
        result['shared'] = len(set_a & set_b)
        result['a_only'] = [
            {'term': t, 'count': int(counts_a[t])}
            for t, _ in counts_a.most_common() if t not in set_b
        ][:top_n]
        result['b_only'] = [
            {'term': t, 'count': int(counts_b[t])}
            for t, _ in counts_b.most_common() if t not in set_a
        ][:top_n]
        result['available'] = bool(result['a_only'] or result['b_only'])
    except Exception as e:
        print(f"Error building competitor keyword gap: {e}")
    return result


def get_rating_distribution(df):
    """Return the 1-5 star distribution of all scraped reviews, overall and
    per platform."""
    result = {'overall': {str(s): 0 for s in range(1, 6)}, 'by_platform': {}}
    if df is None or df.empty or 'score' not in df.columns:
        return result
    try:
        work = df.copy()
        work['score'] = pd.to_numeric(work['score'], errors='coerce').round().clip(1, 5)
        work = work.dropna(subset=['score'])
        work['score'] = work['score'].astype(int)

        overall_counts = work['score'].value_counts()
        result['overall'] = {str(s): int(overall_counts.get(s, 0)) for s in range(1, 6)}

        if 'platform' in work.columns:
            for platform, group in work.groupby('platform'):
                counts = group['score'].value_counts()
                result['by_platform'][str(platform)] = {
                    str(s): int(counts.get(s, 0)) for s in range(1, 6)
                }
    except Exception as e:
        print(f"Error calculating rating distribution: {e}")
    return result


def categorize_complaints(df, lang_code='en'):
    """Group complaints into themes by keyword match.

    A review can count toward several themes. Returns a list ordered by
    priority. lang_code adds that language's keywords to the search.
    """
    themes = []
    if df is None or df.empty or 'review' not in df.columns:
        return themes
    try:
        theme_keywords = get_theme_keywords(lang_code)
        counts = {theme: 0 for theme in theme_keywords}
        likes = {theme: 0 for theme in theme_keywords}
        score_sums = {theme: 0.0 for theme in theme_keywords}

        work = df.dropna(subset=['review']).copy()
        # itertuples renames columns starting with an underscore to positional
        # names, so use a plain one
        work['text_lower'] = work['review'].astype(str).str.lower()
        has_likes = 'likes' in work.columns
        has_score = 'score' in work.columns
        if has_likes:
            work['likes'] = pd.to_numeric(work['likes'], errors='coerce').fillna(0)
        if has_score:
            work['score'] = pd.to_numeric(work['score'], errors='coerce')

        for row in work.itertuples(index=False):
            text = row.text_lower
            row_likes = int(getattr(row, 'likes', 0) or 0) if has_likes else 0
            row_score = getattr(row, 'score', None) if has_score else None
            for theme, keywords in theme_keywords.items():
                if any(kw in text for kw in keywords):
                    counts[theme] += 1
                    likes[theme] += row_likes
                    if row_score is not None and not pd.isna(row_score):
                        score_sums[theme] += float(row_score)

        total = len(work) or 1
        themes = []
        for theme, cnt in counts.items():
            if cnt == 0:
                continue
            avg_score = round(score_sums[theme] / cnt, 2) if has_score and score_sums[theme] else None
            # priority = affected users x severity
            #   affected = complaints + likes (a like is another "me too")
            #   severity = 5 - average rating (lower rating, worse problem)
            affected = cnt + likes[theme]
            severity = (5 - avg_score) if avg_score is not None else 3.0
            themes.append({
                'theme': theme,
                'count': cnt,
                'percent': round(cnt * 100 / total, 1),
                'likes': likes[theme],
                'avg_score': avg_score,
                'affected': affected,
                'priority_raw': round(affected * severity, 2),
            })

        # Normalise to 0-100 against the top theme, for display
        max_priority = max((t['priority_raw'] for t in themes), default=0)
        for t in themes:
            t['priority'] = round(t['priority_raw'] * 100 / max_priority) if max_priority else 0

        themes.sort(key=lambda x: x['priority_raw'], reverse=True)
    except Exception as e:
        print(f"Error categorizing complaints: {e}")
    return themes


def get_platform_comparison(df):
    """Compare average rating, sentiment and volume per platform, across the
    scraped reviews only."""
    comparison = []
    if df is None or df.empty or 'platform' not in df.columns:
        return comparison
    try:
        work = df.copy()
        work['score'] = pd.to_numeric(work['score'], errors='coerce')
        if 'sentiment' not in work.columns:
            work['sentiment'] = work['review'].apply(get_sentiment)
        for platform, group in work.groupby('platform'):
            comparison.append({
                'platform': str(platform),
                'count': int(len(group)),
                'avg_score': round(float(group['score'].mean()), 2) if not group['score'].isna().all() else 0.0,
                'avg_sentiment': round(float(group['sentiment'].mean()), 2) if not group['sentiment'].isna().all() else 0.0,
            })
    except Exception as e:
        print(f"Error building platform comparison: {e}")
    return comparison


def get_emerging_keywords(df, top_n=10):
    """Find keywords rising in the newer half of the complaints.

    Complaints are split chronologically into an older and a newer window. In
    each window a word's document frequency is divided by that window's
    complaint count, so windows of different sizes compare fairly. Returns
    words ordered by the increase in percentage points.
    """
    result = {'keywords': [], 'recent_count': 0, 'previous_count': 0, 'split_date': None}
    if df is None or df.empty or 'cleaned_review' not in df.columns or 'date' not in df.columns:
        return result
    try:
        work = df.dropna(subset=['cleaned_review']).copy()
        work['dt'] = pd.to_datetime(work['date'], errors='coerce')
        work = work.dropna(subset=['dt']).sort_values('dt')

        n = len(work)
        if n < 6:  # too little data for a meaningful trend
            return result

        mid = n // 2
        older = work.iloc[:mid]
        recent = work.iloc[mid:]

        def doc_freq(subdf):
            counts = Counter()
            for text in subdf['cleaned_review']:
                for word in set(str(text).split()):
                    counts[word] += 1
            return counts

        older_c = doc_freq(older)
        recent_c = doc_freq(recent)
        n_old = len(older) or 1
        n_rec = len(recent) or 1

        scored = []
        for word in set(recent_c) | set(older_c):
            if len(word) < 3:
                continue
            rec_hits = recent_c.get(word, 0)
            old_hits = older_c.get(word, 0)
            rec_pct = rec_hits * 100 / n_rec
            old_pct = old_hits * 100 / n_old
            delta = rec_pct - old_pct
            # Only rising words that appear at least twice in the newer window
            if rec_hits >= 2 and delta > 0:
                scored.append({
                    'word': word,
                    'recent_pct': round(rec_pct, 1),
                    'previous_pct': round(old_pct, 1),
                    'change': round(delta, 1),
                    'is_new': old_hits == 0,
                })

        scored.sort(key=lambda x: x['change'], reverse=True)
        result['keywords'] = scored[:top_n]
        result['recent_count'] = int(n_rec)
        result['previous_count'] = int(n_old)
        result['split_date'] = recent['dt'].iloc[0].strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error computing emerging keywords: {e}")
    return result


def filter_by_date_range(df, start_date=None, end_date=None):
    """Filter reviews to a date range; both bounds are optional.

    Expects 'YYYY-MM-DD' strings. Malformed input leaves the data untouched.
    """
    if df is None or df.empty or 'date' not in df.columns:
        return df
    if not start_date and not end_date:
        return df
    try:
        work = df.copy()
        dt = pd.to_datetime(work['date'], errors='coerce')
        # Strip any timezone so tz-aware values compare against naive bounds
        try:
            dt = dt.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass

        mask = dt.notna()
        if start_date:
            mask &= dt >= pd.to_datetime(start_date)
        if end_date:
            # Include the whole end day
            end_inclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= dt <= end_inclusive
        return work[mask]
    except Exception as e:
        print(f"Error filtering by date range: {e}")
        return df


def get_version_breakdown(df, top_n=8):
    """Group complaints by app version, with count, average rating and
    sentiment, to spot which release caused a spike. Google Play only.
    """
    result = []
    if df is None or df.empty or 'version' not in df.columns:
        return result
    try:
        work = df.copy()
        work = work[work['version'].notna()]
        work['version'] = work['version'].astype(str).str.strip()
        work = work[work['version'] != '']
        if work.empty:
            return result

        work['score'] = pd.to_numeric(work['score'], errors='coerce')
        if 'sentiment' not in work.columns:
            work['sentiment'] = work['review'].apply(get_sentiment)

        grouped = work.groupby('version').agg(
            complaints=('review', 'count'),
            avg_score=('score', 'mean'),
            avg_sentiment=('sentiment', 'mean'),
        ).reset_index()
        grouped['complaints'] = grouped['complaints'].astype(int)
        grouped['avg_score'] = grouped['avg_score'].astype(float).round(2)
        grouped['avg_sentiment'] = grouped['avg_sentiment'].astype(float).round(2)

        grouped = grouped.sort_values('complaints', ascending=False).head(top_n)
        result = grouped.to_dict(orient='records')
    except Exception as e:
        print(f"Error computing version breakdown: {e}")
    return result


# ---------------------------------------------------------------------------
# Multilingual sentiment
#
# TextBlob only works in English; elsewhere it returns 0.0 for every review,
# which renders as "neutral" and is silently wrong. The lexicons below cover
# clearly polar vocabulary common in app reviews. For unsupported languages no
# sentiment is computed and the UI says so.
# ---------------------------------------------------------------------------

SENTIMENT_LEXICONS = {
    'tr': {
        # negative
        'berbat': -1.0, 'rezalet': -1.0, 'korkunç': -1.0, 'iğrenç': -1.0, 'felaket': -1.0,
        'kötü': -0.7, 'saçma': -0.7, 'bozuk': -0.7, 'yavaş': -0.6, 'donuyor': -0.7,
        'çöküyor': -0.8, 'çöktü': -0.8, 'kasıyor': -0.6, 'hatalı': -0.6, 'sorunlu': -0.6,
        'çalışmıyor': -0.8, 'açılmıyor': -0.8, 'pahalı': -0.5, 'gereksiz': -0.5,
        'dolandırıcı': -1.0, 'vasat': -0.4, 'yetersiz': -0.5, 'sinir': -0.7, 'nefret': -0.9,
        # positive
        'harika': 1.0, 'mükemmel': 1.0, 'muhteşem': 1.0, 'süper': 0.8, 'güzel': 0.6,
        'başarılı': 0.7, 'hızlı': 0.6, 'kolay': 0.5, 'faydalı': 0.6, 'bayıldım': 0.9,
        'sevdim': 0.8, 'memnun': 0.7, 'teşekkür': 0.5, 'iyi': 0.5,
    },
    'de': {
        'schrecklich': -1.0, 'furchtbar': -1.0, 'katastrophe': -1.0, 'katastrophal': -1.0,
        'schlecht': -0.7, 'mies': -0.8, 'langsam': -0.6, 'absturz': -0.8, 'stürzt': -0.8,
        'fehler': -0.5, 'nervig': -0.7, 'ärgerlich': -0.7, 'unbrauchbar': -0.9,
        'teuer': -0.5, 'betrug': -1.0, 'nutzlos': -0.8, 'enttäuscht': -0.7, 'hasse': -0.9,
        'super': 0.8, 'toll': 0.8, 'großartig': 1.0, 'ausgezeichnet': 1.0, 'perfekt': 1.0,
        'gut': 0.5, 'schnell': 0.6, 'einfach': 0.4, 'hilfreich': 0.6, 'liebe': 0.8,
        'empfehlenswert': 0.7, 'zufrieden': 0.7,
    },
    'es': {
        'terrible': -1.0, 'horrible': -1.0, 'pésimo': -1.0, 'basura': -0.9,
        'malo': -0.7, 'mala': -0.7, 'lento': -0.6, 'lenta': -0.6, 'falla': -0.6,
        'error': -0.5, 'problema': -0.5, 'molesto': -0.6, 'caro': -0.5, 'estafa': -1.0,
        'inútil': -0.8, 'decepcionado': -0.7, 'odio': -0.9,
        'excelente': 1.0, 'genial': 0.9, 'maravilloso': 1.0, 'perfecto': 1.0,
        'bueno': 0.5, 'buena': 0.5, 'rápido': 0.6, 'fácil': 0.5, 'útil': 0.6,
        'encanta': 0.9, 'recomiendo': 0.7,
    },
    'fr': {
        'horrible': -1.0, 'terrible': -1.0, 'catastrophe': -1.0, 'nul': -0.8,
        'mauvais': -0.7, 'mauvaise': -0.7, 'lent': -0.6, 'lente': -0.6, 'plante': -0.8,
        'erreur': -0.5, 'problème': -0.5, 'pénible': -0.7, 'cher': -0.5, 'arnaque': -1.0,
        'inutile': -0.8, 'déçu': -0.7, 'déteste': -0.9,
        'excellent': 1.0, 'génial': 0.9, 'parfait': 1.0, 'merveilleux': 1.0,
        'bon': 0.5, 'bonne': 0.5, 'rapide': 0.6, 'facile': 0.5, 'utile': 0.6,
        'adore': 0.9, 'recommande': 0.7,
    },
}

# Negations flip the polarity of a matched word
NEGATION_WORDS = {
    'tr': {'değil', 'yok', 'hiç'},
    'de': {'nicht', 'kein', 'keine', 'nie', 'niemals'},
    'es': {'no', 'nunca', 'nada', 'ni'},
    'fr': {'ne', 'pas', 'jamais', 'aucun', 'aucune'},
}

# English via TextBlob, the rest via lexicon
SUPPORTED_SENTIMENT_LANGS = {'en'} | set(SENTIMENT_LEXICONS)

_TOKEN_RE = re.compile(r"[^\w']+", re.UNICODE)


def is_sentiment_supported(lang_code):
    """Can sentiment be computed for this language?"""
    return (lang_code or 'en').lower() in SUPPORTED_SENTIMENT_LANGS


def _lexicon_sentiment(text, lang_code):
    """Lexicon-based polarity. A negation within the preceding two tokens
    flips the sign of the matched word."""
    lexicon = SENTIMENT_LEXICONS.get(lang_code, {})
    negations = NEGATION_WORDS.get(lang_code, set())
    tokens = [t for t in _TOKEN_RE.split(text.lower()) if t]

    scores = []
    for i, token in enumerate(tokens):
        polarity = lexicon.get(token)
        if polarity is None:
            # Prefix match to catch inflected forms; risky on short words, so min 5 chars
            for entry, value in lexicon.items():
                if len(entry) >= 5 and token.startswith(entry):
                    polarity = value
                    break
        if polarity is None:
            continue
        window = tokens[max(0, i - 2):i]
        if any(w in negations for w in window):
            polarity = -polarity
        scores.append(polarity)

    if not scores:
        return 0.0
    return max(-1.0, min(1.0, sum(scores) / len(scores)))


def get_sentiment(text, lang_code='en'):
    """Return a review's polarity in [-1, 1].

    TextBlob for English, lexicon-based for the other supported languages.
    Unsupported languages return 0.0 — so that this is not mistaken for
    "neutral", the UI hides its sentiment sections (see is_sentiment_supported).
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        lang = (lang_code or 'en').lower()
        if lang in SENTIMENT_LEXICONS:
            return _lexicon_sentiment(text, lang)
        if lang == 'en':
            return TextBlob(text).sentiment.polarity
        return 0.0
    except Exception as e:
        print(f"Could not process sentiment for a review. Error: {e}")
        return 0.0

def preprocess_and_filter_complaints(df, threshold, app_name, lang_code, extra_stopwords_str=""):
    if 'score' not in df.columns:
        return None
        
    complaints_df = df[df['score'] <= threshold].copy()
    if complaints_df.empty: 
        return None

    complaints_df['sentiment'] = complaints_df['review'].apply(
        lambda text: get_sentiment(text, lang_code)
    )
    # Carry the analysis language to the later, language-aware steps
    complaints_df.attrs['lang_code'] = lang_code

    lang_map = {'en': 'english', 'tr': 'turkish', 'de': 'german',
                'es': 'spanish', 'fr': 'french'}
    nltk_lang = lang_map.get(lang_code, 'english')
    
    stop_words_list = list(stopwords.words(nltk_lang))
    
    if extra_stopwords_str:
        user_stopwords = [word.strip().lower() for word in extra_stopwords_str.split(',') if word.strip()]
        stop_words_list.extend(user_stopwords)
    
    stop_words_list.append(app_name.lower())
    stop_words_list = list(set(stop_words_list))

    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return " ".join([word for word in text.split() if word not in stop_words_list])

    complaints_df['cleaned_review'] = complaints_df['review'].apply(clean_text)
    return complaints_df

def analyze_and_visualize(df, top_n, lang_code=None):
    # Fall back to the language attached during preprocessing
    if lang_code is None:
        lang_code = df.attrs.get('lang_code', 'en')

    all_complaint_text = " ".join(df['cleaned_review'].dropna())
    
    most_common_words = None
    image_data = None
    if all_complaint_text.strip():
        word_counts = Counter(all_complaint_text.split())
        most_common_words = word_counts.most_common(top_n)
        wordcloud = WordCloud(width=800, height=500, background_color='white', colormap='viridis', collocations=False).generate(all_complaint_text)
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        image_data = base64.b64encode(img.getvalue()).decode('utf-8')

    sentiment_summary = {
        'positive': int((df['sentiment'] > 0.05).sum()),
        'negative': int((df['sentiment'] < -0.05).sum()),
        'neutral': int(((df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)).sum()),
        'average_polarity': float(df['sentiment'].mean()) if not pd.isna(df['sentiment'].mean()) else 0.0
    }

    trend_data = []
    try:
        df_trend = df.copy()
        df_trend['date_str'] = pd.to_datetime(df_trend['date']).dt.strftime('%Y-%m-%d')
        trend_grouped = df_trend.groupby('date_str').agg(
            avg_score=('score', 'mean'),
            avg_sentiment=('sentiment', 'mean'),
            count=('review', 'count')
        ).sort_index().reset_index()
        # Cast numpy types to plain Python so JSON serialisation cannot break
        trend_grouped['avg_score'] = trend_grouped['avg_score'].astype(float).round(2)
        trend_grouped['avg_sentiment'] = trend_grouped['avg_sentiment'].astype(float).round(2)
        trend_grouped['count'] = trend_grouped['count'].astype(int)
        trend_data = trend_grouped.to_dict(orient='records')
    except Exception as e:
        print(f"Error calculating trend: {e}")

    sample_reviews = []
    try:
        sample_df = df.copy()
        # Most-endorsed complaints first: likes show how many users hit the
        # same problem
        if 'likes' in sample_df.columns:
            sample_df['likes'] = pd.to_numeric(sample_df['likes'], errors='coerce').fillna(0).astype(int)
            sample_df = sample_df.sort_values('likes', ascending=False)
        sample_df = sample_df.head(50)
        sample_df['sentiment'] = sample_df['sentiment'].round(2)
        cols = ['review', 'score', 'sentiment']
        if 'platform' in sample_df.columns:
            cols.append('platform')
        if 'likes' in sample_df.columns:
            cols.append('likes')
        sample_reviews = sample_df[cols].to_dict(orient='records')
    except Exception as e:
        print(f"Error preparing sample reviews: {e}")

    # Themes, including the analysis language's keywords
    themes = categorize_complaints(df, lang_code)

    # Rising complaint keywords
    emerging_keywords = get_emerging_keywords(df)

    # Complaints per app version
    version_breakdown = get_version_breakdown(df)

    return {
        'most_common_words': most_common_words,
        'image_data': image_data,
        'sentiment_summary': sentiment_summary,
        'trend_data': trend_data,
        'sample_reviews': sample_reviews,
        'themes': themes,
        'emerging_keywords': emerging_keywords,
        'version_breakdown': version_breakdown,
        'sentiment_available': is_sentiment_supported(lang_code),
        'lang_code': lang_code,
    }


# ---------------------------------------------------------------------------
# ASO (App Store Optimization)
#
# Compares the store listing against the language users actually write in.
# The valuable output is the keyword gap: terms users repeat in reviews that
# the listing never mentions.
# ---------------------------------------------------------------------------

# Store character limits
ASO_LIMITS = {
    'google': {'title': 30, 'description': 4000},
    'ios': {'title': 30, 'description': 4000},
}

# Words with no search intent. Without these, the keyword gap fills up with
# filler like "getting", "keeps", "ever", "thank".
ASO_GENERIC_WORDS = {
    # general
    'app', 'apps', 'application', 'free', 'new', 'best', 'one', 'way', 'thing',
    'things', 'stuff', 'lot', 'bit', 'kind', 'sort', 'part', 'time', 'times',
    'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    # verbs and their inflections
    'get', 'gets', 'getting', 'got', 'use', 'uses', 'using', 'used', 'make',
    'makes', 'making', 'made', 'go', 'goes', 'going', 'went', 'do', 'does',
    'doing', 'did', 'know', 'knows', 'think', 'thinks', 'say', 'says', 'said',
    'see', 'sees', 'look', 'looks', 'come', 'comes', 'take', 'takes', 'give',
    'gives', 'put', 'let', 'want', 'wants', 'need', 'needs', 'try', 'tries',
    'trying', 'tell', 'feel', 'feels', 'keep', 'keeps', 'keeping', 'play',
    'plays', 'playing', 'add', 'adds', 'added', 'thank', 'thanks', 'hope',
    # adjective and adverb filler
    'like', 'just', 'also', 'can', 'will', 'really', 'much', 'many', 'good',
    'great', 'please', 'even', 'every', 'always', 'never', 'still', 'back',
    'since', 'without', 'everything', 'something', 'anything', 'nothing',
    'someone', 'people', 'guys', 'ever', 'always', 'maybe', 'actually',
    'pretty', 'quite', 'very', 'super', 'awesome', 'nice', 'amazing', 'love',
    'hate', 'bad', 'worst', 'better', 'well', 'yet', 'though', 'however',
    # Saf degerlendirme sifatlari: POS etiketleyici bunlari baglamsiz kullanimda
    # bazen isim sanir, ama hicbir arama niyeti tasimazlar
    'excellent', 'cool', 'perfect', 'fantastic', 'wonderful', 'terrible',
    'horrible', 'useless', 'annoying', 'disappointing', 'poor', 'decent',
    'okay', 'fine', 'ok', 'meh', 'garbage', 'trash', 'crap', 'stupid',
}

# Bigrams carry a more specific signal than single words ("offline mode",
# "sound quality"), so they get a small ranking boost.
ASO_BIGRAM_BOOST = 1.6


def _aso_tokens(text, lang_code='en'):
    """Split text into the words that carry meaning for ASO."""
    if not isinstance(text, str) or not text.strip():
        return []
    lang_map = {'en': 'english', 'tr': 'turkish', 'de': 'german',
                'es': 'spanish', 'fr': 'french'}
    try:
        stop = set(stopwords.words(lang_map.get((lang_code or 'en').lower(), 'english')))
    except Exception:
        stop = set()
    stop |= ASO_GENERIC_WORDS

    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    return [w for w in words if len(w) >= 3 and not w.isdigit() and w not in stop]


def _noun_set(text, lang_code='en'):
    """POS-tag raw text and return the set of words tagged as nouns.

    Tagging must run on the RAW text: in a stopword-stripped bag the tagger
    has no context and becomes unreliable. Measured on such a bag,
    "audiobooks" came back a verb (VBP) and "wherever" a noun (NN).
    """
    if (lang_code or 'en').lower() != 'en' or not text:
        return None  # None = "filtreleme yapma"
    try:
        import nltk
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        return {w for w, tag in nltk.pos_tag(words) if tag.startswith('NN')}
    except Exception:
        return None  # etiketleyici yoksa sessizce filtresiz devam et


def _noun_filter(tokens, lang_code='en', context_text=None):
    """Reduce a token list to its nouns.

    With context_text the tagging runs on that raw text, which is accurate.
    Without it the tokens themselves are tagged, which loses context.
    """
    if (lang_code or 'en').lower() != 'en' or not tokens:
        return tokens
    nouns = _noun_set(context_text if context_text is not None else ' '.join(tokens), lang_code)
    if nouns is None:
        return tokens
    return [t for t in tokens if t in nouns]


def _metadata_health(store_key, stats):
    """Score one store's title/description length and rating health."""
    limits = ASO_LIMITS[store_key]
    title = (stats.get('title') or '').strip()
    description = stats.get('description_full') or ''
    rating = stats.get('rating') or 0
    review_count = stats.get('reviews') or 0

    title_len = len(title)
    desc_len = len(description)

    def status(ok, warn):
        return 'good' if ok else ('warn' if warn else 'bad')

    return {
        'store': 'Google Play' if store_key == 'google' else 'App Store',
        'title': title,
        'title_length': title_len,
        'title_limit': limits['title'],
        # A title well under the limit leaves valuable keyword space unused
        'title_status': status(title_len >= limits['title'] * 0.7, title_len > 0),
        'description_length': desc_len,
        'description_limit': limits['description'],
        'description_status': status(desc_len >= limits['description'] * 0.5, desc_len > 500),
        'rating': round(float(rating), 2),
        'rating_status': status(rating >= 4.0, rating >= 3.5),
        'reviews': int(review_count),
        'reviews_status': status(review_count >= 1000, review_count >= 100),
    }


def get_aso_report(summary_stats, all_reviews, lang_code='en', top_n=15):
    """Build an ASO report comparing the listing against user vocabulary.

    - metadata: title/description length, rating and review-volume health
    - keyword_opportunities: terms users use that the listing never mentions
    - listing_keywords: what the listing already targets
    """
    report = {
        'metadata': [],
        'keyword_opportunities': [],
        'listing_keywords': [],
        'reviews_analyzed': 0,
        'available': False,
    }

    try:
        stats = summary_stats or {}
        # --- Metadata health ---
        for key in ('google', 'ios'):
            if stats.get(key):
                report['metadata'].append(_metadata_health(key, stats[key]))

        # --- Listeleme metnini topla ---
        listing_parts = []
        for key in ('google', 'ios'):
            if stats.get(key):
                listing_parts.append(stats[key].get('title') or '')
                listing_parts.append(stats[key].get('description_full') or '')
        listing_text = ' '.join(listing_parts).lower()

        if listing_text.strip():
            listing_tokens = _aso_tokens(listing_text, lang_code)
            listing_counts = Counter(listing_tokens)
            report['listing_keywords'] = [
                {'term': term, 'count': int(count)}
                for term, count in listing_counts.most_common(top_n)
            ]

        # --- Keyword gap: user vocabulary vs the listing ---
        if all_reviews is not None and not all_reviews.empty and 'review' in all_reviews.columns:
            texts = all_reviews['review'].dropna().astype(str).tolist()
            report['reviews_analyzed'] = len(texts)

            listing_tokens_all = _aso_tokens(listing_text, lang_code)
            listing_word_set = set(listing_tokens_all)
            # Exclude phrases the listing already uses
            listing_phrase_set = {
                f'{a} {b}' for a, b in zip(listing_tokens_all, listing_tokens_all[1:])
            }
            # Exclude the app's own name; its brand is not an opportunity
            for key in ('google', 'ios'):
                if stats.get(key):
                    listing_word_set |= set(_aso_tokens(stats[key].get('title') or '', lang_code))

            unigram_freq = Counter()
            bigram_freq = Counter()
            for text in texts:
                tokens = _aso_tokens(text, lang_code)
                # Tag the raw review text so the tagger keeps its context
                nouns = set(_noun_filter(tokens, lang_code, context_text=text))
                unigram_freq.update(nouns)
                # Phrases come from the unfiltered stream so adjective+noun
                # patterns like "sound quality" survive, but must contain at
                # least one noun, otherwise pairs like "worse worse" leak in.
                bigram_freq.update({
                    f'{a} {b}' for a, b in zip(tokens, tokens[1:])
                    if a in nouns or b in nouns
                })

            candidates = []
            for term, mentions in unigram_freq.items():
                if term in listing_word_set or mentions < 2:
                    continue
                candidates.append((term, mentions, mentions, False))
            for phrase, mentions in bigram_freq.items():
                if phrase in listing_phrase_set or mentions < 2:
                    continue
                # A phrase whose words all appear in the listing adds nothing
                if all(w in listing_word_set for w in phrase.split()):
                    continue
                candidates.append((phrase, mentions, mentions * ASO_BIGRAM_BOOST, True))

            candidates.sort(key=lambda c: c[2], reverse=True)
            report['keyword_opportunities'] = [{
                'term': term,
                'mentions': int(mentions),
                'percent': round(mentions * 100 / len(texts), 1),
                'is_phrase': is_phrase,
            } for term, mentions, _score, is_phrase in candidates[:top_n]]

        report['available'] = bool(report['metadata'] or report['keyword_opportunities'])
    except Exception as e:
        print(f"Error building ASO report: {e}")

    return report


def build_app_report(google_id, apple_name, country, language, max_reviews,
                     complaint_threshold, top_words, extra_stopwords_str="",
                     start_date=None, end_date=None, force_refresh=False):
    """Run the full analysis for one app and return a consolidated report.

    Shared by the /analyze, /compare and /compare-countries routes.
    start_date/end_date filter the reviews; force_refresh skips the cache.
    """
    report = {
        'name': apple_name or google_id or 'Unknown App',
        'summary_stats': {'google': None, 'ios': None},
        'total_reviews': 0,
        'complaint_count': 0,
        'rating_distribution': None,
        'platform_comparison': [],
        'most_common_words': None,
        'image_data': None,
        'sentiment_summary': None,
        'trend_data': [],
        'sample_reviews': [],
        'themes': [],
        'emerging_keywords': {'keywords': [], 'recent_count': 0, 'previous_count': 0, 'split_date': None},
        'version_breakdown': [],
        'sentiment_available': is_sentiment_supported(language),
        'lang_code': (language or 'en').lower(),
        'aso': {'metadata': [], 'keyword_opportunities': [], 'listing_keywords': [],
                'reviews_analyzed': 0, 'available': False},
        'momentum': {'platforms': [], 'available': False},
        'responses': {'available': False, 'total': 0, 'replied': 0, 'response_rate': 0.0,
                      'complaints_total': 0, 'complaints_replied': 0,
                      'complaint_response_rate': 0.0, 'median_hours': None, 'by_theme': []},
        'error': None,
    }

    all_reviews, summary_stats = fetch_reviews_cached(
        google_id, apple_name, country, language, max_reviews,
        ttl=0 if force_refresh else None,
    )
    report['summary_stats'] = summary_stats

    # Prefer the store title as the display name
    if summary_stats.get('google'):
        report['name'] = summary_stats['google'].get('title') or report['name']
    elif summary_stats.get('ios'):
        report['name'] = summary_stats['ios'].get('title') or report['name']

    if all_reviews.empty:
        report['error'] = "No reviews could be found or fetched for this app."
        return report

    # Optional date-range filter
    all_reviews = filter_by_date_range(all_reviews, start_date, end_date)
    if all_reviews.empty:
        report['error'] = "No reviews found within the selected date range."
        return report

    report['total_reviews'] = int(len(all_reviews))
    report['rating_distribution'] = get_rating_distribution(all_reviews)
    report['platform_comparison'] = get_platform_comparison(all_reviews)
    # ASO and momentum ignore the complaint filter and use all reviews
    report['aso'] = get_aso_report(summary_stats, all_reviews, language)
    report['momentum'] = get_momentum(summary_stats, all_reviews)
    report['responses'] = get_response_analysis(all_reviews, language)

    complaints = preprocess_and_filter_complaints(
        all_reviews, complaint_threshold, apple_name, language, extra_stopwords_str
    )
    if complaints is None or complaints.empty:
        report['error'] = "No complaint reviews were found in the specified score range."
        return report

    report['complaint_count'] = int(len(complaints))
    analysis = analyze_and_visualize(complaints, top_words, lang_code=language)
    report.update(analysis)
    return report
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
import requests # EKLENDİ: Apple iTunes API'den meta veri çekmek için
import urllib.parse

# Import TextBlob for sentiment analysis
from textblob import TextBlob

from google_play_scraper import reviews, Sort, app as google_app_details
from app_store_scraper import AppStore

# Dosyanın en üstüne import kısmına ekleyin:
import urllib.parse

# 1. fetch_reviews_store Fonksiyonunu şu şekilde güncelleyin:
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
            # Google Play yorumları uygulama sürümünü içerir (varsa koruyoruz)
            if 'reviewCreatedVersion' in google_df.columns:
                cols.append('reviewCreatedVersion')
                rename['reviewCreatedVersion'] = 'version'
            google_df_renamed = google_df[cols].rename(columns=rename)
            google_df_renamed['platform'] = 'Google Play'
            if 'version' not in google_df_renamed.columns:
                google_df_renamed['version'] = None
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
            'description': gp_details.get('description', '')[:200] + '...' if gp_details.get('description') else 'No description'
        }
    except Exception as e:
        print(f"Google Play fetch error->: {e}")

    # --- Apple Store (AYRILMIŞ BLOKLAR) ---
    
    # 1. Önce iTunes API ile Meta Verileri Çek
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
                'description': app_data.get('description', '')[:200] + '...' if app_data.get('description') else 'No description'
            }
    except Exception as e:
        print(f"iTunes API Error ->: {e}")

    # 2. Sonra Yorumları Çek
    try:
        apple_scraper = AppStore(country=country, app_name=apple_name)
        apple_scraper.review(how_many=max_reviews_to_fetch)
        if apple_scraper.reviews:
            apple_df = pd.DataFrame(apple_scraper.reviews)
            apple_df_renamed = apple_df[['review', 'rating', 'date']].rename(columns={'rating': 'score'})
            apple_df_renamed['platform'] = 'App Store'
            apple_df_renamed['version'] = None  # App Store scraper sürüm bilgisi vermiyor
            all_dfs.append(apple_df_renamed)
    except Exception as e:  
        print(f"Apple Store fetch review error ->: {e}")
    
    reviews_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return reviews_df, summary_stats


# ---------------------------------------------------------------------------
# Scraping sonuçları için TTL cache
#
# Mağaza scraping'i yavaş ve rate-limit'e tabi. Aynı (uygulama, ülke, dil,
# yorum sayısı) kombinasyonu kısa süre içinde tekrar istendiğinde yeniden
# scrape etmek yerine bellekten dönüyoruz. Özellikle çoklu ülke karşılaştırması
# ve ardışık denemelerde ciddi hız kazancı sağlıyor.
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS = 3600   # 1 saat
CACHE_MAX_ENTRIES = 32     # bellek kullanımını sınırlar (LRU tahliye)

_review_cache = OrderedDict()  # key -> (timestamp, DataFrame, summary_stats)
_cache_lock = threading.Lock()


def clear_review_cache():
    """Cache'i tamamen boşaltır (testler ve manuel yenileme için)."""
    with _cache_lock:
        _review_cache.clear()


def review_cache_info():
    """Cache durumunu döndürür."""
    with _cache_lock:
        return {
            'entries': len(_review_cache),
            'ttl_seconds': CACHE_TTL_SECONDS,
            'max_entries': CACHE_MAX_ENTRIES,
        }


def fetch_reviews_cached(google_id, apple_name, country, lang,
                         max_reviews_to_fetch, ttl=None):
    """fetch_reviews_store'un TTL cache'li sarmalayıcısı.

    ttl=0 verilirse cache atlanır (zorunlu yenileme). Boş sonuçlar cache'lenmez;
    scraper'ın geçici hatası bir saat boyunca sabitlenmesin diye.
    Dönen veriler kopyadır, böylece çağıran taraf cache'i bozamaz.
    """
    ttl = CACHE_TTL_SECONDS if ttl is None else ttl
    key = (google_id, apple_name, country, lang, max_reviews_to_fetch)
    now = time.time()

    with _cache_lock:
        entry = _review_cache.get(key)
        if entry is not None:
            cached_at, cached_df, cached_stats = entry
            if now - cached_at < ttl:
                _review_cache.move_to_end(key)  # LRU: en son kullanılan sona
                return cached_df.copy(), copy.deepcopy(cached_stats)
            del _review_cache[key]  # süresi dolmuş

    reviews_df, summary_stats = fetch_reviews_store(
        google_id, apple_name, country, lang, max_reviews_to_fetch
    )

    if reviews_df is not None and not reviews_df.empty:
        with _cache_lock:
            _review_cache[key] = (time.time(), reviews_df.copy(),
                                  copy.deepcopy(summary_stats))
            _review_cache.move_to_end(key)
            while len(_review_cache) > CACHE_MAX_ENTRIES:
                _review_cache.popitem(last=False)  # en eski kullanılanı at

    return reviews_df, summary_stats


# Şikayet temalarını yakalamak için anahtar kelime -> kategori haritası.
# Her tema, bir yorumda geçtiğinde o temaya sayılan tetikleyici kelimeleri içerir.
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


def get_rating_distribution(df):
    """Çekilen TÜM yorumların yıldız (1-5) dağılımını hem toplam
    hem de platform bazında döndürür. Sensor Tower tarzı bir görünüm sağlar."""
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


def categorize_complaints(df):
    """Şikayet yorumlarını anahtar kelime eşleştirmesiyle temalara ayırır.
    Bir yorum birden fazla temaya sayılabilir. En sık temalardan başlayarak sıralı liste döndürür."""
    themes = []
    if df is None or df.empty or 'review' not in df.columns:
        return themes
    try:
        counts = {theme: 0 for theme in COMPLAINT_THEMES}
        texts = df['review'].dropna().astype(str).str.lower().tolist()
        for text in texts:
            for theme, keywords in COMPLAINT_THEMES.items():
                if any(kw in text for kw in keywords):
                    counts[theme] += 1
        total = len(texts) or 1
        themes = [
            {'theme': theme, 'count': cnt, 'percent': round(cnt * 100 / total, 1)}
            for theme, cnt in counts.items() if cnt > 0
        ]
        themes.sort(key=lambda x: x['count'], reverse=True)
    except Exception as e:
        print(f"Error categorizing complaints: {e}")
    return themes


def get_platform_comparison(df):
    """Google Play ve App Store için ortalama puan, ortalama sentiment ve yorum
    sayısını karşılaştıran özet döndürür (yalnızca çekilen yorumlar üzerinden)."""
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
    """Şikayetleri zaman ekseninde ikiye bölerek (eski yarı vs yeni yarı) son
    dönemde YÜKSELEN kelimeleri tespit eder. Sensor Tower tarzı 'trending complaints'.

    Her pencerede kelimenin 'kaç şikayette geçtiği' (document frequency) o pencerenin
    şikayet sayısına oranlanır; böylece pencere boyutları farklı olsa da adil karşılaştırılır.
    Sonuç: yükseliş miktarına (yüzde puan farkı) göre sıralı kelime listesi."""
    result = {'keywords': [], 'recent_count': 0, 'previous_count': 0, 'split_date': None}
    if df is None or df.empty or 'cleaned_review' not in df.columns or 'date' not in df.columns:
        return result
    try:
        work = df.dropna(subset=['cleaned_review']).copy()
        work['dt'] = pd.to_datetime(work['date'], errors='coerce')
        work = work.dropna(subset=['dt']).sort_values('dt')

        n = len(work)
        if n < 6:  # trend için anlamlı veri yok
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
            # Yalnızca son pencerede en az 2 kez geçen ve yükselen kelimeler
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
    """Yorumları verilen tarih aralığına göre filtreler (her ikisi de opsiyonel).
    start/end 'YYYY-MM-DD' formatında string bekler. Hatalı/eksik girdide veri değişmez."""
    if df is None or df.empty or 'date' not in df.columns:
        return df
    if not start_date and not end_date:
        return df
    try:
        work = df.copy()
        dt = pd.to_datetime(work['date'], errors='coerce')
        # Zaman dilimi bilgisini kaldır (tz-aware ise), böylece naive tarihlerle karşılaştırılabilir
        try:
            dt = dt.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass

        mask = dt.notna()
        if start_date:
            mask &= dt >= pd.to_datetime(start_date)
        if end_date:
            # Bitiş gününün tamamını dahil et
            end_inclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= dt <= end_inclusive
        return work[mask]
    except Exception as e:
        print(f"Error filtering by date range: {e}")
        return df


def get_version_breakdown(df, top_n=8):
    """Şikayetleri uygulama sürümüne göre gruplar; her sürüm için şikayet sayısı,
    ortalama puan ve ortalama sentiment döndürür. Hangi sürümde sorun arttığını
    (regresyon) tespit etmeye yarar. Not: sürüm bilgisi yalnızca Google Play'de mevcut."""
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


def get_sentiment(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Could not process sentiment for a review. Error: {e}")
        return 0.0

def preprocess_and_filter_complaints(df, threshold, app_name, lang_code, extra_stopwords_str=""):
    if 'score' not in df.columns:
        return None
        
    complaints_df = df[df['score'] <= threshold].copy()
    if complaints_df.empty: 
        return None

    complaints_df['sentiment'] = complaints_df['review'].apply(get_sentiment)

    lang_map = {'en': 'english', 'tr': 'turkish'}
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

def analyze_and_visualize(df, top_n):
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
        # Numpy tiplerini açıkça Python tiplerine çeviriyoruz ki JSON serileştirme bozulmasın
        trend_grouped['avg_score'] = trend_grouped['avg_score'].astype(float).round(2)
        trend_grouped['avg_sentiment'] = trend_grouped['avg_sentiment'].astype(float).round(2)
        trend_grouped['count'] = trend_grouped['count'].astype(int)
        trend_data = trend_grouped.to_dict(orient='records')
    except Exception as e:
        print(f"Error calculating trend: {e}")

    sample_reviews = []
    try:
        sample_df = df.copy().head(50)
        sample_df['sentiment'] = sample_df['sentiment'].round(2)
        cols = ['review', 'score', 'sentiment']
        if 'platform' in sample_df.columns:
            cols.append('platform')
        sample_reviews = sample_df[cols].to_dict(orient='records')
    except Exception as e:
        print(f"Error preparing sample reviews: {e}")

    # Şikayet temalarını çıkar
    themes = categorize_complaints(df)

    # Yükselen (trending) şikayet kelimeleri
    emerging_keywords = get_emerging_keywords(df)

    # Sürüm bazlı şikayet dağılımı
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
    }


def build_app_report(google_id, apple_name, country, language, max_reviews,
                     complaint_threshold, top_words, extra_stopwords_str="",
                     start_date=None, end_date=None, force_refresh=False):
    """Tek bir uygulama için uçtan uca analiz çalıştırıp konsolide bir rapor dict'i döndürür.
    Hem /analyze hem de /compare route'ları tarafından kullanılır.
    start_date/end_date verilirse yorumlar bu tarih aralığına göre filtrelenir.
    force_refresh=True ise cache atlanır ve mağazadan yeniden çekilir."""
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
        'error': None,
    }

    all_reviews, summary_stats = fetch_reviews_cached(
        google_id, apple_name, country, language, max_reviews,
        ttl=0 if force_refresh else None,
    )
    report['summary_stats'] = summary_stats

    # Görünen ad olarak mağaza başlığını tercih et
    if summary_stats.get('google'):
        report['name'] = summary_stats['google'].get('title') or report['name']
    elif summary_stats.get('ios'):
        report['name'] = summary_stats['ios'].get('title') or report['name']

    if all_reviews.empty:
        report['error'] = "No reviews could be found or fetched for this app."
        return report

    # Opsiyonel tarih aralığı filtresi
    all_reviews = filter_by_date_range(all_reviews, start_date, end_date)
    if all_reviews.empty:
        report['error'] = "No reviews found within the selected date range."
        return report

    report['total_reviews'] = int(len(all_reviews))
    report['rating_distribution'] = get_rating_distribution(all_reviews)
    report['platform_comparison'] = get_platform_comparison(all_reviews)

    complaints = preprocess_and_filter_complaints(
        all_reviews, complaint_threshold, apple_name, language, extra_stopwords_str
    )
    if complaints is None or complaints.empty:
        report['error'] = "No complaint reviews were found in the specified score range."
        return report

    report['complaint_count'] = int(len(complaints))
    analysis = analyze_and_visualize(complaints, top_words)
    report.update(analysis)
    return report
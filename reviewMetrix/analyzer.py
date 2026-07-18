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
            'description': gp_details.get('description', '')[:200] + '...' if gp_details.get('description') else 'No description',
            # ASO analizi tam metne ihtiyaç duyar; yukarıdaki kısaltılmış alan yalnızca gösterim içindir
            'description_full': gp_details.get('description', '') or '',
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
                'description': app_data.get('description', '')[:200] + '...' if app_data.get('description') else 'No description',
                'description_full': app_data.get('description', '') or '',
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

# Aynı temaların diğer dillerdeki karşılıkları. Analiz dili İngilizce dışındaysa
# bu kelimeler İngilizce listeyle BİRLEŞTİRİLİR (yorumlar sık sık İngilizce
# terim de içerdiği için ikisini birden aramak isabeti artırır).
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
    """Verilen dil için tema->anahtar kelime haritasını döndürür.
    İngilizce dışındaki diller için İngilizce liste ile birleştirilir."""
    base = {theme: list(words) for theme, words in COMPLAINT_THEMES.items()}
    extra = THEME_KEYWORDS_BY_LANG.get((lang_code or 'en').lower())
    if extra:
        for theme, words in extra.items():
            base.setdefault(theme, []).extend(words)
    return base


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


def categorize_complaints(df, lang_code='en'):
    """Şikayet yorumlarını anahtar kelime eşleştirmesiyle temalara ayırır.
    Bir yorum birden fazla temaya sayılabilir. En sık temalardan başlayarak sıralı liste döndürür.
    lang_code verilirse o dilin anahtar kelimeleri de aramaya dahil edilir."""
    themes = []
    if df is None or df.empty or 'review' not in df.columns:
        return themes
    try:
        theme_keywords = get_theme_keywords(lang_code)
        counts = {theme: 0 for theme in theme_keywords}
        texts = df['review'].dropna().astype(str).str.lower().tolist()
        for text in texts:
            for theme, keywords in theme_keywords.items():
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


# ---------------------------------------------------------------------------
# Çok dilli sentiment
#
# TextBlob yalnızca İngilizce çalışır; diğer dillerde her yoruma 0.0 (nötr)
# döndürür ki bu sessizce yanlıştır. Aşağıdaki sözlükler uygulama yorumlarında
# sık geçen açıkça olumlu/olumsuz kelimeleri kapsar. Desteklenmeyen dillerde
# sentiment hesaplanmaz ve arayüz bunu açıkça belirtir.
# ---------------------------------------------------------------------------

SENTIMENT_LEXICONS = {
    'tr': {
        # olumsuz
        'berbat': -1.0, 'rezalet': -1.0, 'korkunç': -1.0, 'iğrenç': -1.0, 'felaket': -1.0,
        'kötü': -0.7, 'saçma': -0.7, 'bozuk': -0.7, 'yavaş': -0.6, 'donuyor': -0.7,
        'çöküyor': -0.8, 'çöktü': -0.8, 'kasıyor': -0.6, 'hatalı': -0.6, 'sorunlu': -0.6,
        'çalışmıyor': -0.8, 'açılmıyor': -0.8, 'pahalı': -0.5, 'gereksiz': -0.5,
        'dolandırıcı': -1.0, 'vasat': -0.4, 'yetersiz': -0.5, 'sinir': -0.7, 'nefret': -0.9,
        # olumlu
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

# Olumsuzlama ekleri: eşleşen kelimenin polaritesini ters çevirir
NEGATION_WORDS = {
    'tr': {'değil', 'yok', 'hiç'},
    'de': {'nicht', 'kein', 'keine', 'nie', 'niemals'},
    'es': {'no', 'nunca', 'nada', 'ni'},
    'fr': {'ne', 'pas', 'jamais', 'aucun', 'aucune'},
}

# İngilizce TextBlob ile, diğerleri sözlükle desteklenir
SUPPORTED_SENTIMENT_LANGS = {'en'} | set(SENTIMENT_LEXICONS)

_TOKEN_RE = re.compile(r"[^\w']+", re.UNICODE)


def is_sentiment_supported(lang_code):
    """Verilen dilde sentiment analizi yapılabiliyor mu?"""
    return (lang_code or 'en').lower() in SUPPORTED_SENTIMENT_LANGS


def _lexicon_sentiment(text, lang_code):
    """Sözlük tabanlı polarite. Olumsuzlama kelimesi öncesindeki 2 token içinde
    geçiyorsa eşleşen kelimenin işareti ters çevrilir."""
    lexicon = SENTIMENT_LEXICONS.get(lang_code, {})
    negations = NEGATION_WORDS.get(lang_code, set())
    tokens = [t for t in _TOKEN_RE.split(text.lower()) if t]

    scores = []
    for i, token in enumerate(tokens):
        polarity = lexicon.get(token)
        if polarity is None:
            # Ekli biçimleri yakalamak için önek eşleşmesi (kısa kelimelerde riskli, min 5 harf)
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
    """Yorumun duygu polaritesini [-1, 1] aralığında döndürür.

    İngilizce için TextBlob, desteklenen diğer diller için sözlük tabanlı
    yöntem kullanılır. Desteklenmeyen dillerde 0.0 döner — bu değerin
    "nötr" sanılmaması için arayüzde sentiment bölümleri gizlenir
    (bkz. is_sentiment_supported).
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
    # Analiz dilini sonraki adımlara taşı (tema/sentiment dil duyarlılığı için)
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
    # Dil açıkça verilmediyse preprocess adımında iliştirilen değeri kullan
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

    # Şikayet temalarını çıkar (analiz dilinin anahtar kelimeleri dahil)
    themes = categorize_complaints(df, lang_code)

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
        'sentiment_available': is_sentiment_supported(lang_code),
        'lang_code': lang_code,
    }


# ---------------------------------------------------------------------------
# ASO (App Store Optimization)
#
# Mağaza listeleme metinlerini kullanıcıların gerçekte kullandığı dille
# karşılaştırır. En değerli çıktı "keyword gap": kullanıcıların yorumlarda sık
# kullandığı ama listelemede hiç geçmeyen kelimeler — doğrudan aksiyon alınabilir.
# ---------------------------------------------------------------------------

# Mağaza karakter limitleri (2024 itibarıyla)
ASO_LIMITS = {
    'google': {'title': 30, 'description': 4000},
    'ios': {'title': 30, 'description': 4000},
}

# ASO açısından değersiz genel kelimeler. Yorum metinlerinde çok sık geçen ama
# hiçbir arama niyeti taşımayan fiil/zarf/dolgu kelimeleri elenmezse keyword gap
# listesi "getting, keeps, ever, thank" gibi gürültüyle dolar.
ASO_GENERIC_WORDS = {
    # genel
    'app', 'apps', 'application', 'free', 'new', 'best', 'one', 'way', 'thing',
    'things', 'stuff', 'lot', 'bit', 'kind', 'sort', 'part', 'time', 'times',
    'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    # fiiller ve çekimleri
    'get', 'gets', 'getting', 'got', 'use', 'uses', 'using', 'used', 'make',
    'makes', 'making', 'made', 'go', 'goes', 'going', 'went', 'do', 'does',
    'doing', 'did', 'know', 'knows', 'think', 'thinks', 'say', 'says', 'said',
    'see', 'sees', 'look', 'looks', 'come', 'comes', 'take', 'takes', 'give',
    'gives', 'put', 'let', 'want', 'wants', 'need', 'needs', 'try', 'tries',
    'trying', 'tell', 'feel', 'feels', 'keep', 'keeps', 'keeping', 'play',
    'plays', 'playing', 'add', 'adds', 'added', 'thank', 'thanks', 'hope',
    # sıfat/zarf dolguları
    'like', 'just', 'also', 'can', 'will', 'really', 'much', 'many', 'good',
    'great', 'please', 'even', 'every', 'always', 'never', 'still', 'back',
    'since', 'without', 'everything', 'something', 'anything', 'nothing',
    'someone', 'people', 'guys', 'ever', 'always', 'maybe', 'actually',
    'pretty', 'quite', 'very', 'super', 'awesome', 'nice', 'amazing', 'love',
    'hate', 'bad', 'worst', 'better', 'well', 'yet', 'though', 'however',
}

# Bigram'lar tek kelimelerden daha spesifik ASO sinyali taşır ("offline mode",
# "sound quality"), bu yüzden sıralamada hafifçe öne çıkarılır.
ASO_BIGRAM_BOOST = 1.6


def _aso_tokens(text, lang_code='en'):
    """Metni ASO analizi için anlamlı kelimelere ayırır."""
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


def _noun_filter(tokens, lang_code='en'):
    """ASO anahtar kelimeleri ezici çoğunlukla isimdir ('offline', 'timer'),
    fiil/sıfat değil ('would', 'love', 'broken'). İngilizce için POS etiketleme
    ile isimleri süzer.

    NLTK'nın etiketleyicisi yalnızca İngilizce çalışır; diğer dillerde veya
    etiketleyici yoksa token'lar olduğu gibi döndürülür (stopword filtresi
    yine de uygulanmış olur).
    """
    if (lang_code or 'en').lower() != 'en' or not tokens:
        return tokens
    try:
        import nltk
        return [word for word, tag in nltk.pos_tag(tokens) if tag.startswith('NN')]
    except Exception:
        return tokens  # etiketleyici yoksa sessizce eski davranışa dön


def _metadata_health(store_key, stats):
    """Bir mağaza için başlık/açıklama uzunluğu ve puan sağlığını değerlendirir."""
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
        # Başlık limitin %70'inden kısaysa değerli keyword alanı boşta demektir
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
    """Mağaza listelemesi ile kullanıcı dilini karşılaştıran ASO raporu üretir.

    - metadata: başlık/açıklama uzunluğu, puan ve yorum sayısı sağlığı
    - keyword_opportunities: yorumlarda sık geçen ama listelemede olmayan kelimeler
    - listing_keywords: listelemenin hâlihazırda hedeflediği kelimeler
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
        # --- Metadata sağlığı ---
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

        # --- Keyword gap: kullanıcı dili vs listeleme ---
        if all_reviews is not None and not all_reviews.empty and 'review' in all_reviews.columns:
            texts = all_reviews['review'].dropna().astype(str).tolist()
            report['reviews_analyzed'] = len(texts)

            listing_tokens_all = _aso_tokens(listing_text, lang_code)
            listing_word_set = set(listing_tokens_all)
            # Listelemede geçen ifadeleri de hariç tut
            listing_phrase_set = {
                f'{a} {b}' for a, b in zip(listing_tokens_all, listing_tokens_all[1:])
            }
            # Uygulama adını da hariç tut (kendi markası ASO fırsatı değil)
            for key in ('google', 'ios'):
                if stats.get(key):
                    listing_word_set |= set(_aso_tokens(stats[key].get('title') or '', lang_code))

            unigram_freq = Counter()
            bigram_freq = Counter()
            for text in texts:
                tokens = _aso_tokens(text, lang_code)
                nouns = set(_noun_filter(tokens, lang_code))
                unigram_freq.update(nouns)
                # İfadeler tam akıştan üretilir ki "sound quality" gibi sıfat+isim
                # kalıpları kaybolmasın, ancak en az bir isim içermeleri gerekir —
                # aksi halde "worse worse" gibi anlamsız ikililer listeye sızar.
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
                # Her iki kelimesi de listelemede geçen ifade yeni bilgi taşımaz
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
        'sentiment_available': is_sentiment_supported(language),
        'lang_code': (language or 'en').lower(),
        'aso': {'metadata': [], 'keyword_opportunities': [], 'listing_keywords': [],
                'reviews_analyzed': 0, 'available': False},
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
    # ASO şikayet filtresinden bağımsız: tüm yorumların dilini kullanır
    report['aso'] = get_aso_report(summary_stats, all_reviews, language)

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
# reviewMetrix/analyzer.py
import pandas as pd
from nltk.corpus import stopwords
import re
from collections import Counter
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
            google_df_renamed = google_df[['content', 'score', 'at']].rename(columns={'content': 'review', 'at': 'date'})
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
            all_dfs.append(apple_df_renamed)
    except Exception as e:  
        print(f"Apple Store fetch review error ->: {e}")
    
    reviews_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return reviews_df, summary_stats


# 2. analyze_and_visualize fonksiyonundaki "trend_data" kısmını böyle güncelleyin:
    # Calculate time series trend
    trend_data = []
    try:
        df_trend = df.copy()
        df_trend['date_str'] = pd.to_datetime(df_trend['date']).dt.strftime('%Y-%m-%d')
        trend_grouped = df_trend.groupby('date_str').agg(
            avg_score=('score', 'mean'),
            avg_sentiment=('sentiment', 'mean'),
            count=('review', 'count')
        ).sort_index().reset_index()
        
        # JSON'a çevrilirken bozulmaması için açıkça Python tiplerine (astype) dönüştürüyoruz
        trend_grouped['avg_score'] = trend_grouped['avg_score'].astype(float).round(2)
        trend_grouped['avg_sentiment'] = trend_grouped['avg_sentiment'].astype(float).round(2)
        trend_grouped['count'] = trend_grouped['count'].astype(int)
        
        trend_data = trend_grouped.to_dict(orient='records')
    except Exception as e:
        print(f"Error calculating trend: {e}")

# --- AŞAĞIDAKİ FONKSİYONLAR ORİJİNAL HALİYLE BIRAKILMIŞTIR ---

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
        trend_grouped['avg_score'] = trend_grouped['avg_score'].round(2)
        trend_grouped['avg_sentiment'] = trend_grouped['avg_sentiment'].round(2)
        trend_data = trend_grouped.to_dict(orient='records')
    except Exception as e:
        print(f"Error calculating trend: {e}")

    sample_reviews = []
    try:
        sample_df = df.copy().head(50)
        sample_df['sentiment'] = sample_df['sentiment'].round(2)
        sample_reviews = sample_df[['review', 'score', 'sentiment']].to_dict(orient='records')
    except Exception as e:
        print(f"Error preparing sample reviews: {e}")
    
    return most_common_words, image_data, sentiment_summary, trend_data, sample_reviews
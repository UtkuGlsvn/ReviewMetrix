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

from google_play_scraper import reviews, Sort
from app_store_scraper import AppStore

def fetch_reviews_store(google_id, apple_name, country, lang, max_reviews_to_fetch):
    all_dfs = []
    # Google Play
    try:
        google_reviews_data, _ = reviews(google_id, lang=lang, country=country, sort=Sort.NEWEST, count=max_reviews_to_fetch)
        google_df = pd.DataFrame(google_reviews_data)
        google_df_renamed = google_df[['content', 'score', 'at']].rename(columns={'content': 'review', 'at': 'date'})
        all_dfs.append(google_df_renamed)
    except Exception as e:
        print(f"Google Play fetch review error->: {e}")
    # Apple Store
    try:
        apple_scraper = AppStore(country=country, app_name=apple_name)
        apple_scraper.review(how_many=max_reviews_to_fetch)
        if apple_scraper.reviews:
            apple_df = pd.DataFrame(apple_scraper.reviews)
            apple_df_renamed = apple_df[['review', 'rating', 'date']].rename(columns={'rating': 'score'})
            all_dfs.append(apple_df_renamed)
    except Exception as e:
        print(f"Apple Store fetch review error ->: {e}")
    
    if not all_dfs: return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def preprocess_and_filter_complaints(df, threshold, app_name, lang_code, extra_stopwords_str=""):
    complaints_df = df[df['score'] <= threshold].copy()
    if complaints_df.empty: return None

    lang_map = {'en': 'english', 'tr': 'turkish'}
    nltk_lang = lang_map.get(lang_code, 'english')
    
    stop_words_list = list(stopwords.words(nltk_lang))
    
    if extra_stopwords_str:
        user_stopwords = [word.strip().lower() for word in extra_stopwords_str.split(',') if word.strip()]
        stop_words_list.extend(user_stopwords)
    
    stop_words_list.append(app_name.lower())

    stop_words_list = list(set(stop_words_list))
    
    stop_words_list.append(app_name)

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
    if not all_complaint_text.strip(): return None, None

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
    return most_common_words, image_data

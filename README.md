App Review Analysis Tool (ReviewMetrix)

<p align="center">
  <img src="https://raw.githubusercontent.com/UtkuGlsvn/ReviewMetrix/main/img/ReviewMetrix.png" width="350" alt="ReviewMetrix Main Page">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/UtkuGlsvn/ReviewMetrix/main/img/ReviewMetrixAnalys.png" width="350" alt="ReviewMetrix Analysis Page">
</p>

Description
ReviewMetrix is a powerful web-based tool built with Flask that fetches and analyzes user reviews for any application from the Google Play Store and Apple App Store. It helps developers and product managers quickly identify common complaints and negative feedback by visualizing the most frequent keywords found in low-rated reviews.

Features
Dual-Platform Support: Fetches reviews simultaneously from both Google Play Store and Apple App Store.

Complaint-Focused Analysis: Filters reviews based on a user-defined score threshold (e.g., 2 stars and below) to focus specifically on negative feedback.

Word Cloud Visualization: Generates an intuitive word cloud from complaint reviews, making it easy to spot recurring themes at a glance.

Top Keywords List: Displays a ranked list of the most frequently used words in complaints for detailed analysis.

Fully Interactive Web UI: All parameters, including App IDs, store country, language, and analysis settings, can be configured through a user-friendly web interface.

Customizable Stopwords: Allows users to add their own list of irrelevant words to exclude from the analysis for more accurate results.

Project Structure
The project is organized using a scalable Flask application factory pattern with Blueprints.

/ReviewMetrix/
|
├── reviewMetrix/
|   |
|   ├── __init__.py         # Application factory
|   ├── routes.py           # Handles web routes and logic
|   ├── analyzer.py         # Core data fetching and analysis functions
|   |
|   └── templates/
|       ├── index.html      # Main form page
|       └── results.html    # Results display page
|
└── run.py                  # Main script to run the application

Setup and Installation
Follow these steps to get the application running on your local machine.

1. Prerequisites
Python 3.8+

pip and venv

2. Clone the Repository
git clone [https://github.com/UtkuGlsvn/ReviewMetrix.git](https://github.com/UtkuGlsvn/ReviewMetrix.git)
cd ReviewMetrix

3. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

On Windows:

python -m venv venv
.\venv\Scripts\activate

4. Install Dependencies
First, create a requirements.txt file by running this command in your terminal:

pip freeze > requirements.txt

Note: If you haven't installed the libraries yet, you can create the file manually with the following content: Flask, pandas, nltk, wordcloud, matplotlib, google-play-scraper, app-store-scraper.

Then, install all the required libraries from the requirements.txt file:

pip install -r requirements.txt

The application will also automatically download the necessary stopwords data from NLTK on first run.

5. Run the Application
Once the setup is complete, you can start the Flask server:

python run.py

The application will be available at http://127.0.0.1:5000.

How to Use
Open your web browser and navigate to http://127.0.0.1:5000.

Fill out the form with the required parameters:

Google Play App ID: (e.g., com.google.android.gm for Gmail)

Apple Store App Name: (e.g., gmail-email-by-google for Gmail)

Store Country & Language: Use two-letter codes (e.g., us, en).

Analysis Parameters: Set the number of reviews to fetch, the score threshold for complaints, and any custom stopwords.

Click the "Start Analysis" button.

The results page will display the total number of reviews fetched, the number of complaints analyzed, a list of the most common words, and a word cloud visualization.

Technologies Used
Backend: Python, Flask

Data Analysis: Pandas, NLTK

Visualization: Matplotlib, WordCloud

Web Scraping: google-play-scraper, `app-

import pandas as pd
import numpy as np
import requests
import gzip
import shutil
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Ordner für Daten
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# URLs der Frankfurt-Daten von Inside Airbnb (Stand 2025)
URLS = {
    'listings': 'http://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/listings.csv.gz',
    'reviews': 'http://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/reviews.csv.gz',
    'calendar': 'http://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/calendar.csv.gz'
}

def download_and_extract(name, url):
    local_gz = f'{DATA_DIR}/{name}.csv.gz'
    local_csv = f'{DATA_DIR}/{name}.csv'

    # Herunterladen
    r = requests.get(url, stream=True)
    with open(local_gz, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    print(f"{name}.csv.gz heruntergeladen!")

    # Entpacken
    with gzip.open(local_gz, 'rb') as f_in:
        with open(local_csv, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"{name}.csv entpackt!")

    return local_csv

# Alle Dateien laden
listings_file = download_and_extract('listings', URLS['listings'])
reviews_file  = download_and_extract('reviews', URLS['reviews'])
calendar_file = download_and_extract('calendar', URLS['calendar'])

# Einlesen in Pandas
listings = pd.read_csv(listings_file)
reviews  = pd.read_csv(reviews_file)
calendar = pd.read_csv(calendar_file)

print("Daten geladen!")
print("Listings:", listings.shape)
print("Reviews:", reviews.shape)
print("Calendar:", calendar.shape)

# Grundlegendes Preprocessing
# Preis als float
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)

# Pool vorhanden
listings['has_pool'] = listings['amenities'].fillna('').str.contains('Pool').astype(int)

# Sentiment der Beschreibung
analyzer = SentimentIntensityAnalyzer()
listings['description'] = listings['description'].fillna('')
listings['sentiment'] = listings['description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Kategorie-Encoding für room_type
from sklearn.preprocessing import LabelEncoder
if 'room_type' in listings.columns:
    le = LabelEncoder()
    listings['room_type'] = le.fit_transform(listings['room_type'].astype(str))

# Feature-Auswahl (kannst erweitern)
features = ['accommodates', 'bedrooms', 'bathrooms', 'has_pool', 'sentiment', 'room_type']
X = listings[features]
y = listings['price']

print("Feature-Matrix X:", X.shape)
print("Target y:", y.shape)

# Optional: Speichern für späteres Training
X.to_csv(f'{DATA_DIR}/X_frankfurt.csv', index=False)
y.to_csv(f'{DATA_DIR}/y_frankfurt.csv', index=False)
print("Preprocessed Daten gespeichert!")

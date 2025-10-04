import pandas as pd
import requests
import os
import gzip
import shutil

# Ordner für Daten
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# URLs für Frankfurt-Daten (aktuell 2025-10-01)
URLS = {
    'listings': 'https://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/listings.csv',
    'reviews': 'https://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/reviews.csv',
    'calendar': 'https://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/calendar.csv'
}

def download_file(name, url):
    local_path = f'{DATA_DIR}/{name}.csv'
    r = requests.get(url)
    if r.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(r.content)
        print(f'{name}.csv erfolgreich heruntergeladen!')
    else:
        print(f'Fehler beim Herunterladen von {name}: {r.status_code}')
    return local_path

if __name__ == '__main__':
    listings_file = download_file('listings', URLS['listings'])
    reviews_file = download_file('reviews', URLS['reviews'])
    calendar_file = download_file('calendar', URLS['calendar'])

    # Optional: Daten einlesen, um zu prüfen
    listings = pd.read_csv(listings_file)
    reviews = pd.read_csv(reviews_file)
    calendar = pd.read_csv(calendar_file)

    print("Listings Vorschau:")
    print(listings.head())
    print("Reviews Vorschau:")
    print(reviews.head())
    print("Calendar Vorschau:")
    print(calendar.head())

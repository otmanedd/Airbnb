import pandas as pd
import requests
import os

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    'listings': 'https://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/listings.csv',
    'reviews': 'https://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/reviews.csv',
    'calendar': 'https://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/calendar.csv'
}

for name, url in URLS.items():
    r = requests.get(url)
    with open(f'{DATA_DIR}/{name}.csv', 'wb') as f:
        f.write(r.content)
    print(f'{name} heruntergeladen!')

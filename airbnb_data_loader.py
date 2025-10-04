import pandas as pd
import gzip, shutil, os, requests

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    'listings': 'http://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/listings.csv.gz',
    'reviews': 'http://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/reviews.csv.gz',
    'calendar': 'http://data.insideairbnb.com/germany/hessen/frankfurt-am-main/2025-10-01/data/calendar.csv.gz'
}

def download_and_extract(name, url):
    local_gz = f'{DATA_DIR}/{name}.csv.gz'
    local_csv = f'{DATA_DIR}/{name}.csv'
    r = requests.get(url, stream=True)
    with open(local_gz, 'wb') as f: shutil.copyfileobj(r.raw, f)
    with gzip.open(local_gz, 'rb') as f_in:
        with open(local_csv, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return local_csv

for key in URLS:
    download_and_extract(key, URLS[key])
print("Daten Frankfurt heruntergeladen und entpackt!")

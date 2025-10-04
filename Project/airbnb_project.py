import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Daten laden
listings = pd.read_csv('../data/listings.csv')

# Preprocessing
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
listings['has_pool'] = listings['amenities'].fillna('').str.contains('Pool').astype(int)

# Sentiment der Beschreibung
analyzer = SentimentIntensityAnalyzer()
listings['description'] = listings['description'].fillna('')
listings['sentiment'] = listings['description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Kategorie-Encoding
le = LabelEncoder()
if 'room_type' in listings.columns:
    listings['room_type'] = le.fit_transform(listings['room_type'].astype(str))

# Features und Ziel
features = ['accommodates','bedrooms','bathrooms','has_pool','sentiment','room_type']
X = listings[features]
y = listings['price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Modell speichern
with open('../Deployment/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("Modell trainiert und gespeichert!")

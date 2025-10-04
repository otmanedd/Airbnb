import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLTK setup
nltk.download('punkt')

# Daten einlesen
listings = pd.read_csv('../data/listings.csv')
reviews = pd.read_csv('../data/reviews.csv')
calendar = pd.read_csv('../data/calendar.csv')

# Preprocessing
# Beispiel: nur einige Features, du kannst erweitern
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
listings['has_pool'] = listings['amenities'].str.contains('Pool').astype(int)

# Textfeatures
analyzer = SentimentIntensityAnalyzer()
listings['sentiment'] = listings['description'].fillna('').apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Kategorie-Encoding
for col in ['room_type', 'neighborhood_group']:
    if col in listings.columns:
        le = LabelEncoder()
        listings[col] = le.fit_transform(listings[col].astype(str))

# Features + Target
features = ['accommodates', 'bedrooms', 'bathrooms', 'has_pool', 'sentiment']
X = listings[features]
y = listings['price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Speichern
import pickle
with open('../Deployment/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)


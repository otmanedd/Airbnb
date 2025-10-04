import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Modell laden
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

st.title("Airbnb Preisvorhersage für Frankfurt 🏠")

# Eingabe-Widgets
accommodates = st.number_input('Anzahl Gäste', min_value=1, max_value=20, value=2)
bedrooms = st.number_input('Anzahl Schlafzimmer', min_value=0, max_value=10, value=1)
bathrooms = st.number_input('Anzahl Badezimmer', min_value=0.0, max_value=10.0, value=1.0)
has_pool = st.selectbox('Pool vorhanden?', ['Nein', 'Ja'])
description = st.text_area('Beschreibung der Unterkunft')

# Feature vorbereiten
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores(description)['compound']

has_pool_val = 1 if has_pool == 'Ja' else 0

X_test = pd.DataFrame([[accommodates, bedrooms, bathrooms, has_pool_val, sentiment]],
                      columns=['accommodates','bedrooms','bathrooms','has_pool','sentiment'])

# Vorhersage
if st.button('Preis vorhersagen'):
    pred = xgb_model.predict(X_test)[0]
    st.success(f"Geschätzter Preis: €{round(pred, 2)} pro Nacht")


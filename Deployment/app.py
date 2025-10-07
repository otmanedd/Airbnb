import streamlit as st
import pandas as pd
import numpy as np
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Modell laden
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

st.title("🏠 Airbnb Preisvorhersage Frankfurt")

st.write("Gib unten die Eigenschaften deiner Unterkunft ein, um den geschätzten Preis pro Nacht zu berechnen.")

# Eingabe-Widgets
accommodates = st.number_input('Anzahl Gäste', min_value=1, max_value=20, value=2)
bedrooms = st.number_input('Anzahl Schlafzimmer', min_value=0, max_value=10, value=1)
bathrooms = st.number_input('Anzahl Badezimmer', min_value=0.0, max_value=10.0, value=1.0)
has_pool = st.selectbox('Pool vorhanden?', ['Nein', 'Ja'])
room_type = st.selectbox('Zimmertyp', ['Entire home/apt', 'Private room', 'Shared room'])
description = st.text_area('Beschreibung', "Helle Wohnung im Zentrum von Frankfurt, nah an Restaurants und U-Bahn.")

# Sentiment berechnen (Beschreibung)
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores(description)['compound']

# Eingaben in numerische Werte umwandeln
has_pool_val = 1 if has_pool == 'Ja' else 0
room_type_val = {'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2}[room_type]

# Eingabedaten in DataFrame
X_test = pd.DataFrame([[accommodates, bedrooms, bathrooms, has_pool_val, sentiment, room_type_val]],
                      columns=['accommodates', 'bedrooms', 'bathrooms', 'has_pool', 'sentiment', 'room_type'])

# --- FEATURE-NAME FIX ---
# Prüfen, ob Modell andere Feature-Namen erwartet
try:
    model_features = xgb_model.feature_names_in_
    if list(X_test.columns) != list(model_features):
        # Wenn Spaltennamen anders sind: Daten anpassen (Dummy-Matching)
        X_fixed = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
        for col in X_test.columns:
            if col in X_fixed.columns:
                X_fixed[col] = X_test[col].values
        X_test = X_fixed
except AttributeError:
    # Falls Modell kein feature_names_in_ Attribut hat, einfach weitermachen
    pass

# Vorhersage-Button
if st.button('💰 Preis vorhersagen'):
    try:
        pred = xgb_model.predict(X_test)[0]
        if pred < 0:
            pred = 0
        st.success(f"Geschätzter Preis: **€{round(pred, 2)}** pro Nacht")
    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")

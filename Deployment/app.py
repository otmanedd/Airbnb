import streamlit as st
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Modell laden
with open('xgb_model.pkl','rb') as f:
    xgb_model = pickle.load(f)

st.title("Airbnb Preisvorhersage Frankfurt 🏠")

# Eingabe-Widgets
accommodates = st.number_input('Anzahl Gäste', min_value=1, max_value=20, value=2)
bedrooms = st.number_input('Anzahl Schlafzimmer', min_value=0, max_value=10, value=1)
bathrooms = st.number_input('Anzahl Badezimmer', min_value=0.0, max_value=10.0, value=1.0)
has_pool = st.selectbox('Pool vorhanden?', ['Nein','Ja'])
room_type = st.selectbox('Zimmertyp', ['Entire home/apt','Private room','Shared room'])
description = st.text_area('Beschreibung')

# Features vorbereiten
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores(description)['compound']
has_pool_val = 1 if has_pool == 'Ja' else 0
room_type_val = {'Entire home/apt':0,'Private room':1,'Shared room':2}[room_type]

X_test = pd.DataFrame([[accommodates, bedrooms, bathrooms, has_pool_val, sentiment, room_type_val]],
                      columns=['accommodates','bedrooms','bathrooms','has_pool','sentiment','room_type'])

# Vorhersage
if st.button('Preis vorhersagen'):
    pred = xgb_model.predict(X_test)[0]
    st.success(f"Geschätzter Preis: €{round(pred,2)} pro Nacht")

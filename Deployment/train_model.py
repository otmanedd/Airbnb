import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Beispielhafte Trainingsdaten (simulierte Airbnb-Daten)
data = {
    'accommodates': np.random.randint(1, 10, 500),
    'bedrooms': np.random.randint(0, 5, 500),
    'bathrooms': np.random.uniform(1, 3, 500),
    'has_pool': np.random.randint(0, 2, 500),
    'sentiment': np.random.uniform(-1, 1, 500),
    'room_type': np.random.choice([0, 1, 2], 500),
    'price': np.random.uniform(30, 400, 500)
}
df = pd.DataFrame(data)

# Features und Ziel definieren
X = df[['accommodates', 'bedrooms', 'bathrooms', 'has_pool', 'sentiment', 'room_type']]
y = df['price']

# Splitten und trainieren
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Speichern
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Neues Modell erfolgreich trainiert und gespeichert: xgb_model.pkl")

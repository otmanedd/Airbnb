import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# 🔹 Dummy-Daten (für Frankfurt Airbnb-Beispiele)
data = {
    'accommodates': np.random.randint(1, 6, 300),
    'bedrooms': np.random.randint(0, 3, 300),
    'bathrooms': np.random.uniform(1, 2, 300),
    'has_pool': np.random.randint(0, 2, 300),
    'sentiment': np.random.uniform(-1, 1, 300),
    'room_type': np.random.choice([0, 1, 2], 300),
    'price': np.random.uniform(50, 300, 300)
}
df = pd.DataFrame(data)

X = df[['accommodates', 'bedrooms', 'bathrooms', 'has_pool', 'sentiment', 'room_type']]
y = df['price']

# 🔹 Modell trainieren
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 🔹 Modell speichern
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Neues Modell wurde erfolgreich trainiert und gespeichert: xgb_model.pkl")

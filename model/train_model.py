# model/train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load cleaned dataset
df = pd.read_csv('House Price India.csv')

# Define features and target
features = ['number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
            'number of floors', 'waterfront present', 'number of views', 'condition of the house']
X = df[features]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")

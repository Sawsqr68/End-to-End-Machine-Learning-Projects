"""
Script to train and save IRIS classification model
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('./data/iris.csv', header=None, 
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Prepare features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2%}")

# Save model
joblib.dump(model, './data/model.sav')
print("Model saved to ./data/model.sav")

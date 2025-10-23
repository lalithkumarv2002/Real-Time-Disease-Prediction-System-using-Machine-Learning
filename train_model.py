import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# ---- Sample dataset (you can expand this) ----
data = {
    'fever': [1, 1, 0, 0, 1, 0, 0, 1],
    'cough': [1, 1, 0, 1, 0, 0, 1, 1],
    'fatigue': [1, 0, 1, 0, 1, 1, 0, 1],
    'headache': [1, 0, 1, 0, 1, 0, 1, 1],
    'body_pain': [1, 0, 1, 1, 0, 0, 1, 1],
    'disease': [
        'Flu', 'COVID', 'Migraine', 'Allergy', 
        'Typhoid', 'Diabetes', 'Migraine', 'Flu'
    ]
}

df = pd.DataFrame(data)

X = df.drop('disease', axis=1)
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train model ----
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print("Accuracy:", model.score(X_test, y_test))

# ---- Save model ----
joblib.dump(model, "disease_model.joblib")
print("💾 Model saved as disease_model.joblib")

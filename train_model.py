import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("asl_landmarks.csv")

# Features
X = df.drop("label", axis=1)

# Labels
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

print("Training model...")

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc * 100:.2f}%")

# Save
joblib.dump(model, "asl_model.pkl")

print("Model saved successfully.")
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# 1️⃣ Load your final processed training data
df = pd.read_csv("data/processed/final_training_data.csv")

# 2️⃣ Drop the target column if it exists
if "label" in df.columns:
    X = df.drop(columns=["label"])
else:
    X = df.copy()

# 3️⃣ Convert all categorical columns to numeric using LabelEncoder
for col in X.columns:
    if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# 4️⃣ Fit scaler and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)

# 5️⃣ Save the artifacts
os.makedirs("data/models", exist_ok=True)
joblib.dump(scaler, "data/models/scaler.pkl")
joblib.dump(pca, "data/models/pca.pkl")

print("✅ Saved scaler.pkl and pca.pkl successfully in data/models/")
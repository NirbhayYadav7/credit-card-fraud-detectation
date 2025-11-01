import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)
import joblib

# ===== 1. Paths =====
DATA_PATH = 'creditCardFraud.csv'  # your uploaded dataset
MODEL_DIR = os.path.join('model')
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== 2. Load Data =====
print('Loading dataset...')
df = pd.read_csv(DATA_PATH)

# ===== 3. Identify Target Column =====
TARGET = 'default payment next month'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset")

# ===== 4. Feature Selection =====
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.")

# ===== 5. Split Data =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ===== 6. Preprocessing Pipelines =====
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, [c for c in numeric_cols if c in X.columns]),
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder='drop'
)

# ===== 7. Model Definition =====
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', clf)
])

# ===== 8. Train =====
print('Training model...')
model_pipeline.fit(X_train, y_train)
print('✅ Training complete.')

# ===== 9. Evaluate =====
y_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test)

roc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

print(f'ROC AUC: {roc:.4f}')
print(f'Average precision (PR AUC): {ap:.4f}')
print('Classification report:')
print(classification_report(y_test, y_pred))

# ===== 10. Save Model =====
model_path = os.path.join(MODEL_DIR, 'credit_default_model.pkl')
joblib.dump(model_pipeline, model_path)
print('💾 Model saved to', model_path)

# ===== 11. Save Metadata =====
meta = {
    'created_at': datetime.utcnow().isoformat() + 'Z',
    'roc_auc': float(roc),
    'average_precision': float(ap),
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0])
}

with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print('🧾 Metadata saved successfully.')

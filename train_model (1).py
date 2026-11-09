# train_model.py
# Student Performance Prediction System - data retrieval, preprocessing, training, evaluation
import io
import json
import zipfile
import urllib.request

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Part 1 - Retrieve the UCI Student Performance dataset (ID 320)
# ---------------------------------------------------------------------------
MATH_ROWS = 395  # student-mat.csv has 395 records (student-por.csv has 649)


def load_math_from_zip():
    """Fallback: read student-mat.csv straight from the UCI download archive."""
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    raw = urllib.request.urlopen(url, timeout=60).read()

    def find_math(blob):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                if name.endswith("student-mat.csv"):
                    return pd.read_csv(z.open(name), sep=";")
                if name.endswith(".zip"):  # archive may be nested
                    found = find_math(z.read(name))
                    if found is not None:
                        return found
        return None

    return find_math(raw)


try:
    from ucimlrepo import fetch_ucirepo

    student = fetch_ucirepo(id=320)
    df = pd.concat([student.data.features, student.data.targets], axis=1)
except Exception as exc:  # network / package problem
    print("ucimlrepo failed (", exc, ") - using the UCI zip file instead.")
    df = load_math_from_zip()

# The lab requires the MATHEMATICS records. If the repository interface returned
# a different file (e.g. the 649-row Portuguese file), use the Mathematics file.
if len(df) != MATH_ROWS:
    print(f"Got {len(df)} rows, expected {MATH_ROWS} (Mathematics). Loading student-mat.csv ...")
    df = load_math_from_zip()

df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
assert {"G1", "G2", "G3"}.issubset(df.columns), "G1, G2 and G3 must be in the dataset"

# ---------------------------------------------------------------------------
# Part 2 - Preprocess, train and evaluate
# ---------------------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, pos_label=1, average="binary", zero_division=0
)

print("Accuracy:", round(accuracy, 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))
print("Confusion matrix [rows = actual, cols = predicted] (Not at risk, At risk):")
print(confusion_matrix(y_test, predictions))
print(
    f"At-risk class -> precision: {precision:.3f} | recall: {recall:.3f} | F1: {f1:.3f}"
)

# Keep the numbers for the report / README
with open("model_metrics.json", "w") as f:
    json.dump({
        "rows_used": int(len(df)),
        "test_size": int(len(y_test)),
        "accuracy": round(float(accuracy), 3),
        "at_risk_precision": round(float(precision), 3),
        "at_risk_recall": round(float(recall), 3),
        "at_risk_f1": round(float(f1), 3),
    }, f, indent=2)

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")

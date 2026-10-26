"""Train a leakage-aware model from the official UCI Bank Marketing dataset."""
from pathlib import Path
import io, zipfile, urllib.request
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT = Path(__file__).parent / "model.joblib"

def main():
    print("Downloading UCI Bank Marketing dataset...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(archive.read("bank.zip")))
    csv_name = next(n for n in inner.namelist() if n.endswith("bank-full.csv"))
    df = pd.read_csv(io.BytesIO(inner.read(csv_name)), sep=";")
    X = df.drop(columns=["y", "duration"], errors="ignore")
    y = (df["y"].astype(str).str.lower() == "yes").astype(int)
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("preprocessor", prep), ("model", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    ))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= .5).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "test_samples": len(y_test),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }
    print("Held-out test metrics (threshold=0.5)")
    print(metrics)
    joblib.dump({"model": model, "features": list(X.columns), "metrics": metrics}, OUT)
    print(f"Saved model: {OUT}")
if __name__ == "__main__":
    main()

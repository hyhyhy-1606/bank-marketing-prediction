"""Train a leakage-aware model from the official UCI Bank Marketing dataset."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SEED = 42
TARGET = "y"
LEAKAGE_COLUMN = "duration"
TEST_SIZE = 0.2
DEFAULT_CONFIG = "rf-150-leaf5"

MODEL_CONFIGS = {
    "rf-150-leaf5": lambda: RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "rf-300-leaf2": lambda: RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "hist-gb": lambda: HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED,
    ),
}


def load_dataset():
    """Download the official UCI archive and return the dataframe plus provenance."""
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(archive.read("bank.zip")))
    csv_name = next(n for n in inner.namelist() if n.endswith("bank-full.csv"))
    raw = inner.read(csv_name)
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    metadata = {
        "source": URL,
        "file": csv_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return df, metadata


def build_xy(df):
    X = df.drop(columns=[TARGET, LEAKAGE_COLUMN], errors="ignore")
    y = (df[TARGET].astype(str).str.lower() == "yes").astype(int)
    return X, y


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )


def make_pipeline(model_name, X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", MODEL_CONFIGS[model_name]()),
    ])


def metrics_at_threshold(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "test_samples": int(len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def train_and_save(config_name, out_path, df=None, dataset_metadata=None):
    if df is None:
        df, dataset_metadata = load_dataset()
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = make_pipeline(config_name, X)
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = metrics_at_threshold(y_test, probs)

    bundle = {
        "model": pipeline,
        "features": list(X.columns),
        "metrics": metrics,
        "config": {
            "name": config_name,
            "params": pipeline.named_steps["model"].get_params(),
        },
        "dataset": dataset_metadata,
        "runtime": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, out_path, compress=3)
    metrics["model_size_mb"] = round(Path(out_path).stat().st_size / 1e6, 1)
    return metrics, bundle, (X_test, y_test, probs)


def main():
    out = Path(__file__).parent / "model.joblib"
    print("Downloading UCI Bank Marketing dataset...")
    metrics, bundle, _ = train_and_save(DEFAULT_CONFIG, out)
    print(f"Config: {DEFAULT_CONFIG}")
    print(f"Dataset: {bundle['dataset']['rows']} rows x {bundle['dataset']['columns']} cols "
          f"(sha256 {bundle['dataset']['sha256'][:12]}...)")
    print("Held-out test metrics (threshold=0.5)")
    print(metrics)
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()

"""Offline evaluation for the UCI Bank Marketing model.

Produces the tables used in the report: cross-validation, model comparison,
baselines, bootstrap confidence intervals, threshold/profit analysis, lift by
decile and calibration. Outputs metrics.json plus markdown/CSV tables.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from train_model import (
    MODEL_CONFIGS,
    SEED,
    build_xy,
    load_dataset,
    make_pipeline,
    metrics_at_threshold,
    split_data,
)

C_CALL = 25_000          # VND per call
V_DEPOSIT = 400_000      # VND net profit per converted customer
BOOTSTRAP_ITERS = 1_000
OUT_DIR = Path(__file__).parent / "reports"


def pct(value):
    return f"{value * 100:.2f}%"


def vnd(value):
    return f"{value:,.0f}".replace(",", ".")


def threshold_metrics(y, probs, tau):
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tau": round(float(tau), 2),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "calls": int(tp + fp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "profit": int(tp * V_DEPOSIT - (tp + fp) * C_CALL),
    }


def bootstrap_ci(y, probs, pred, iters=BOOTSTRAP_ITERS, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    samples = {"roc_auc": [], "pr_auc": [], "f1": [], "recall": []}
    for _ in range(iters):
        draw = rng.choice(idx, size=len(idx), replace=True)
        y_s, prob_s, pred_s = y[draw], probs[draw], pred[draw]
        if y_s.sum() in (0, len(y_s)):
            continue
        samples["roc_auc"].append(roc_auc_score(y_s, prob_s))
        samples["pr_auc"].append(average_precision_score(y_s, prob_s))
        samples["f1"].append(f1_score(y_s, pred_s, zero_division=0))
        samples["recall"].append(recall_score(y_s, pred_s, zero_division=0))
    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in samples.items()
    }


def lift_table(y, probs, bins=10):
    order = np.argsort(-probs)
    y_sorted = y[order]
    base_rate = float(y.mean())
    total_pos = int(y.sum())
    rows, cumulative = [], 0
    for rank, chunk in enumerate(np.array_split(np.arange(len(y)), bins), start=1):
        positives = int(y_sorted[chunk].sum())
        cumulative += positives
        rate = positives / len(chunk)
        rows.append({
            "decile": rank,
            "n": int(len(chunk)),
            "positives": positives,
            "conversion_rate": rate,
            "lift": rate / base_rate,
            "cumulative_gain": cumulative / total_pos,
        })
    return rows, base_rate


def calibration_table(y, probs, bins=10):
    frame = pd.DataFrame({"prob": probs, "y": y})
    frame["bin"] = pd.qcut(frame["prob"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    rows = [{
        "bin": str(index),
        "n": int(len(group)),
        "mean_predicted": float(group["prob"].mean()),
        "observed": float(group["y"].mean()),
    } for index, group in grouped]
    return rows


def build_report(dataset, cv_table, comparison, baselines, ci, thresholds,
                 best_profit, best_f1, lift_rows, base_rate, calibration,
                 brier, final_config):
    lines = []
    add = lines.append

    add("# Bảng số liệu đánh giá — Bank Marketing\n")
    add("Sinh tự động bởi `backend/evaluate.py`. Không sửa tay; chạy lại script nếu "
        "cấu hình mô hình thay đổi.\n")
    add("## 1. Nguồn dữ liệu\n")
    add("| Thuộc tính | Giá trị |")
    add("| :--- | :--- |")
    add(f"| Tệp | `{dataset['file']}` (UCI, phân cách `;`) |")
    add(f"| Kích thước | {dataset['rows']:,} dòng × {dataset['columns']} cột |".replace(",", "."))
    add(f"| SHA-256 | `{dataset['sha256']}` |")
    add(f"| Mô hình được chọn | `{final_config}` |\n")

    add("## 2. Kiểm định chéo 5-fold (Stratified, shuffle=True)\n")
    add("| Cấu hình | ROC-AUC | PR-AUC | Recall | F1 | Precision | Accuracy |")
    add("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in cv_table:
        add("| {config} | {roc_auc} | {pr_auc} | {recall} | {f1} | {precision} | {accuracy} |".format(**row))

    add("\n## 3. So sánh trên tập test độc lập (9.043 mẫu, ngưỡng 0,5)\n")
    add("| Cấu hình | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Dung lượng | Thời gian fit |")
    add("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in comparison:
        add("| {name} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {pr_auc} | "
            "{size} | {fit_time} |".format(**row))

    add("\n## 4. Baseline tham chiếu\n")
    add("| Baseline | Accuracy | Precision | Recall | F1 | Ghi chú |")
    add("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for row in baselines:
        add("| {name} | {accuracy} | {precision} | {recall} | {f1} | {note} |".format(**row))

    add("\n## 5. Khoảng tin cậy bootstrap 95% (1.000 lần lấy mẫu lại)\n")
    add("| Chỉ số | Giá trị | 95% CI |")
    add("| :--- | :---: | :---: |")
    for key, label, point in [
        ("roc_auc", "ROC-AUC", ci["point"]["roc_auc"]),
        ("pr_auc", "PR-AUC", ci["point"]["pr_auc"]),
        ("f1", "F1", ci["point"]["f1"]),
        ("recall", "Recall", ci["point"]["recall"]),
    ]:
        low, high = ci["intervals"][key]
        add(f"| {label} | {point:.4f} | [{low:.4f}; {high:.4f}] |")

    add("\n## 6. Phân tích ngưỡng và lợi nhuận\n")
    add(f"Giả định: C = {vnd(C_CALL)} VND/cuộc gọi, V = {vnd(V_DEPOSIT)} VND/khách đăng ký. "
        f"Điểm hoà vốn precision = C/V = {pct(C_CALL / V_DEPOSIT)}.\n")
    add("| Ngưỡng τ | Cuộc gọi | TP | FP | Precision | Recall | F1 | Lợi nhuận (VND) |")
    add("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in thresholds:
        add("| {tau:.2f} | {calls} | {tp} | {fp} | {precision} | {recall} | {f1} | {profit} |".format(
            tau=row["tau"], calls=row["calls"], tp=row["tp"], fp=row["fp"],
            precision=pct(row["precision"]), recall=pct(row["recall"]),
            f1=pct(row["f1"]), profit=vnd(row["profit"])))
    add(f"\n- Ngưỡng tối đa hoá lợi nhuận: **τ* = {best_profit['tau']:.2f}** → "
        f"{vnd(best_profit['profit'])} VND, {best_profit['calls']} cuộc gọi, "
        f"Recall {pct(best_profit['recall'])}, Precision {pct(best_profit['precision'])}.")
    add(f"- Ngưỡng tối đa hoá F1: **τ = {best_f1['tau']:.2f}** → F1 {pct(best_f1['f1'])}, "
        f"lợi nhuận {vnd(best_f1['profit'])} VND.\n")

    add("## 7. Lift theo decile\n")
    add(f"Tỷ lệ chuyển đổi nền của tập test: {pct(base_rate)}.\n")
    add("| Decile | Số khách | Chuyển đổi | Tỷ lệ | Lift | Gain tích luỹ |")
    add("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in lift_rows:
        add("| {decile} | {n} | {positives} | {rate} | {lift:.2f}× | {gain} |".format(
            decile=row["decile"], n=row["n"], positives=row["positives"],
            rate=pct(row["conversion_rate"]), lift=row["lift"],
            gain=pct(row["cumulative_gain"])))

    add("\n## 8. Hiệu chuẩn xác suất\n")
    add(f"Brier score = {brier:.4f} (càng thấp càng tốt; mô hình có "
        "`class_weight=\"balanced\"` nên xác suất đầu ra bị đẩy lên và không được "
        "diễn giải như xác suất thực tế).\n")
    add("| Khoảng xác suất dự đoán | Số mẫu | Xác suất TB | Tỷ lệ thực tế |")
    add("| :--- | :---: | :---: | :---: |")
    for row in calibration:
        add(f"| {row['bin']} | {row['n']} | {pct(row['mean_predicted'])} | {pct(row['observed'])} |")

    add("\n---\n")
    add("Sinh lại bằng: `cd backend && python evaluate.py`\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", default=None,
                        help="Cấu hình dùng để sinh metrics.json (mặc định: PR-AUC cao nhất)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print("Đang tải dữ liệu UCI...")
    df, dataset = load_dataset()
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test)
    print(f"Dữ liệu: {dataset['rows']} dòng × {dataset['columns']} cột | "
          f"test = {len(y_test_np)} mẫu | tỷ lệ yes = {y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    cv_table, cv_means, comparison = [], {}, []
    fitted = {}
    for name in MODEL_CONFIGS:
        print(f"[CV] {name} ...")
        pipeline = make_pipeline(name, X)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        cv_means[name] = {
            key: float(scores[f"test_{key}"].mean()) for key in scoring
        }
        cv_table.append({
            "config": name,
            "roc_auc": f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
            "pr_auc": f"{scores['test_pr_auc'].mean():.4f} ± {scores['test_pr_auc'].std():.4f}",
            "recall": f"{scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}",
            "precision": f"{scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}",
            "f1": f"{scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}",
            "accuracy": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
        })

    for name in MODEL_CONFIGS:
        print(f"[TEST] {name} ...")
        pipeline = make_pipeline(name, X)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        probs = pipeline.predict_proba(X_test)[:, 1]
        base = metrics_at_threshold(y_test, probs)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
            tmp_path = Path(handle.name)
        joblib.dump(pipeline, tmp_path, compress=3)
        size_mb = round(tmp_path.stat().st_size / 1e6, 1)
        tmp_path.unlink(missing_ok=True)
        comparison.append({
            "name": name,
            "accuracy": pct(base["accuracy"]),
            "precision": pct(base["precision"]),
            "recall": pct(base["recall"]),
            "f1": pct(base["f1"]),
            "roc_auc": f"{base['roc_auc']:.4f}",
            "pr_auc": f"{average_precision_score(y_test, probs):.4f}",
            "size": f"{size_mb} MB",
            "fit_time": f"{fit_time:.1f}s",
        })
        fitted[name] = {
            "probs": probs,
            "metrics": base,
            "pr_auc": float(average_precision_score(y_test, probs)),
            "size_mb": size_mb,
        }
        del pipeline

    final_config = args.final or max(fitted, key=lambda k: fitted[k]["pr_auc"])
    chosen = fitted[final_config]
    print(f"Cấu hình được chọn: {final_config}")

    baselines = []
    for name, strategy, note in [
        ("Dummy (prior)", "prior", "Luôn dự đoán lớp đa số (no)"),
        ("Call everyone", None, "Gọi toàn bộ khách, không lọc"),
    ]:
        if strategy:
            dummy = DummyClassifier(strategy=strategy, random_state=SEED)
            dummy.fit(X_train, y_train)
            pred = dummy.predict(X_test)
            baselines.append({
                "name": name,
                "accuracy": pct(accuracy_score(y_test, pred)),
                "precision": pct(precision_score(y_test, pred, zero_division=0)),
                "recall": pct(recall_score(y_test, pred, zero_division=0)),
                "f1": pct(f1_score(y_test, pred, zero_division=0)),
                "note": note,
            })
        else:
            positives = int(y_test_np.sum())
            negatives = len(y_test_np) - positives
            baselines.append({
                "name": name,
                "accuracy": pct(positives / len(y_test_np)),
                "precision": pct(positives / len(y_test_np)),
                "recall": "100.00%",
                "f1": pct(2 * positives / (2 * positives + negatives)),
                "note": f"Chi phí {vnd(negatives * C_CALL)} VND, lợi nhuận "
                        f"{vnd(positives * V_DEPOSIT - len(y_test_np) * C_CALL)} VND",
            })

    probs = chosen["probs"]
    pred = (probs >= 0.5).astype(int)
    ci_intervals = bootstrap_ci(y_test_np, probs, pred)
    ci = {
        "point": {
            "roc_auc": chosen["metrics"]["roc_auc"],
            "pr_auc": chosen["pr_auc"],
            "f1": chosen["metrics"]["f1"],
            "recall": chosen["metrics"]["recall"],
        },
        "intervals": ci_intervals,
    }

    grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    full_grid = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]
    full_rows = [threshold_metrics(y_test_np, probs, tau) for tau in full_grid]
    thresholds = [threshold_metrics(y_test_np, probs, tau) for tau in grid]
    best_profit = max(full_rows, key=lambda row: row["profit"])
    best_f1 = max(full_rows, key=lambda row: row["f1"])

    lift_rows, base_rate = lift_table(y_test_np, probs)
    calibration = calibration_table(y_test_np, probs)
    brier = float(brier_score_loss(y_test_np, probs))

    report = build_report(dataset, cv_table, comparison, baselines, ci, thresholds,
                          best_profit, best_f1, lift_rows, base_rate, calibration,
                          brier, final_config)
    (OUT_DIR / "eval_tables.md").write_text(report, encoding="utf-8")

    pd.DataFrame(full_rows).to_csv(OUT_DIR / "threshold_curve.csv", index=False)
    pd.DataFrame(lift_rows).to_csv(OUT_DIR / "lift_table.csv", index=False)
    pd.DataFrame(comparison).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    metrics = dict(chosen["metrics"])
    metrics.update({
        "pr_auc": round(chosen["pr_auc"], 4),
        "brier": round(brier, 4),
        "model_size_mb": chosen["size_mb"],
        "config": final_config,
        "dataset": dataset,
        "recommended_threshold": best_profit["tau"],
        "metrics_at_recommended_threshold": {
            key: best_profit[key] for key in
            ["tau", "precision", "recall", "f1", "calls", "profit", "tp", "fp", "fn", "tn"]
        },
        "economics": {
            "cost_per_call_vnd": C_CALL,
            "value_per_deposit_vnd": V_DEPOSIT,
            "break_even_precision": round(C_CALL / V_DEPOSIT, 4),
        },
        "cv": {
            "folds": 5,
            "scores": {k: round(v, 4) for k, v in cv_means[final_config].items()},
        },
        "bootstrap_ci": {k: list(v) for k, v in ci_intervals.items()},
    })
    (Path(__file__).parent / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== So sánh (test, ngưỡng 0.5) ===")
    for row in comparison:
        print(f"  {row['name']:<14} Acc {row['accuracy']:>7} | Rec {row['recall']:>7} | "
              f"PR-AUC {row['pr_auc']} | AUC {row['roc_auc']} | {row['size']}")
    print(f"\nĐề xuất: {final_config}")
    print(f"Ngưỡng tối ưu lợi nhuận: {best_profit['tau']:.2f} → "
          f"{vnd(best_profit['profit'])} VND, {best_profit['calls']} cuộc gọi, "
          f"Recall {pct(best_profit['recall'])}, Precision {pct(best_profit['precision'])}")
    print(f"Lợi nhuận tại 0.5: {vnd(threshold_metrics(y_test_np, probs, 0.5)['profit'])} VND")
    print(f"Đã ghi: {OUT_DIR / 'eval_tables.md'}")
    print(f"Đã ghi: {OUT_DIR / 'threshold_curve.csv'}, lift_table.csv, model_comparison.csv")
    print(f"Đã ghi: {Path(__file__).parent / 'metrics.json'}")


if __name__ == "__main__":
    main()

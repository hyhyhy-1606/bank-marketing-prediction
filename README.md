# Bank Marketing Analytics — Vercel starter

Live demo: https://bank-marketing-prediction-project.vercel.app/ · API docs: https://bank-marketing-prediction-1o9v.onrender.com/docs

A responsive React + Vite dashboard based on the uploaded notebook's Bank Marketing workflow:
- EDA and campaign summaries
- CSV upload and data explorer
- Subscription outcome, job, age-group and month charts
- Prediction form connected to a separate FastAPI service
- Responsive layout for desktop and mobile

## Important data note
The initial view uses a tiny **illustrative demo dataset**, not the official 45,211-row dataset and not research findings. Upload `bank-full.csv` directly: PapaParse detects the semicolon separator of the original UCI file automatically.

The website derives displayed KPIs from the currently loaded rows. Do not report demo values as experimental results.

## Run locally
Requires Node.js 20+ recommended.

```bash
npm install
npm run dev
```

## Deploy frontend to Vercel
1. Upload this folder to a new GitHub repository.
2. In Vercel, choose **Add New → Project** and import that repository.
3. Framework preset: **Vite**. Build command: `npm run build`; output directory: `dist`.
4. Deploy. `vercel.json` includes an SPA rewrite.
5. For real predictions, set environment variable `VITE_API_BASE_URL` to the deployed backend URL and redeploy.

## Prediction API (separate deployment)
Vercel hosts the frontend. The included Python model API should be deployed separately (for example, on Render).

```bash
cd backend
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
python train_model.py
uvicorn main:app --host 0.0.0.0 --port 8000
```

The training script downloads the official UCI Bank Marketing data, excludes `duration` to avoid post-call leakage, uses a stratified 80/20 split and prints test metrics. The API loads `backend/model.joblib`. Keep the backend's Python runtime compatible with `requirements.txt`. Configure `CORS_ORIGINS` to your deployed Vercel domain for production.

## Evaluation
```bash
cd backend
python evaluate.py
```

This script runs 5-fold stratified cross-validation for three candidate models, compares them on the held-out test set, adds baselines, bootstrap confidence intervals, a threshold/profit curve, decile lift and calibration, then writes:

- `backend/reports/eval_tables.md` — every table used in the report, generated from code
- `backend/reports/threshold_curve.csv`, `lift_table.csv`, `model_comparison.csv`
- `backend/metrics.json` — extended metrics that `/metrics` merges over `model.joblib`

The canonical model is `rf-150-leaf5` (Random Forest, 150 trees, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42`), which reproduces the notebook's held-out results.

## Suggested report upgrades
- Include architecture: UCI CSV → preprocessing/model pipeline → FastAPI inference API → Vercel React dashboard.
- Record the actual test metrics from your run; do not copy example metrics from elsewhere.
- Discuss class imbalance, threshold selection, data leakage (`duration`), and limitations.
- Add screenshots of the deployed site and API `/health` endpoint as evidence.

## Current limitations
- The web dashboard starts with illustrative demo rows and accepts a CSV with a `y` column.
- The API form supplies `default="no"` because the current form does not ask for that field. If you want full feature control, add a default-status field.
- `class_weight="balanced"` inflates predicted probabilities, so they should not be read as calibrated subscription rates.
- The deployed predictor classifies at threshold 0.5. `evaluate.py` also reports a profit-optimal threshold for campaign list selection.

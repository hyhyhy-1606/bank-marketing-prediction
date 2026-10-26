# Bank Marketing Analytics — Vercel starter

A responsive React + Vite dashboard based on the uploaded notebook's Bank Marketing workflow:
- EDA and campaign summaries
- CSV upload and data explorer
- Subscription outcome, job, age-group and month charts
- Prediction form connected to a separate FastAPI service
- Responsive layout for desktop and mobile

## Important data note
The initial view uses a tiny **illustrative demo dataset**, not the official 45,211-row dataset and not research findings. Upload `bank-full.csv` (separator is handled by standard CSV parsing only if exported as comma-delimited; the original UCI file is semicolon-delimited). For the original semicolon-delimited file, open it in Excel/Sheets and export as comma-delimited CSV first, or modify the Papa Parse delimiter setting to `;`.

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

## Suggested report upgrades
- Include architecture: UCI CSV → preprocessing/model pipeline → FastAPI inference API → Vercel React dashboard.
- Record the actual test metrics from your run; do not copy example metrics from elsewhere.
- Discuss class imbalance, threshold selection, data leakage (`duration`), and limitations.
- Add screenshots of the deployed site and API `/health` endpoint as evidence.

## Current limitations
- The web dashboard starts with illustrative demo rows and accepts a CSV with a `y` column.
- Original UCI `bank-full.csv` uses semicolon delimiters. Convert to comma-separated CSV before upload, or update the parser to use `delimiter: ";"`.
- The API form supplies `default="no"` because the current form does not ask for that field. If you want full feature control, add a default-status field.
- This is a starter project, not yet deployed to your Vercel account.

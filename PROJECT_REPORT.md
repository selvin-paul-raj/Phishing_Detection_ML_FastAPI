# Phishing Detection ML API — Project Report

## 1. Executive Summary
This project implements a machine-learning-powered Phishing URL Detection system with a FastAPI backend and a Streamlit user interface. A Random Forest classifier (serialized as `models/RFC_best_model.pkl`) is used to predict whether a provided URL is legitimate or a phishing attempt based on extracted URL features. The system is designed for experimentation, local use, and containerized deployment.

Key deliverables:
- `app/app.py` — FastAPI application exposing the prediction endpoint and health check
- `app/appUI.py` — Streamlit front-end for interactive URL checks
- `models/model_training.py` and `models/model_training_notebook.ipynb` — scripts/notebook for training and reproducing the model
- `models/RFC_best_model.pkl` — trained Random Forest model used for inference
- `data/Website Phishing.csv` — dataset used for model training and evaluation
- `requirements.txt` — Python dependencies


## 2. Objectives
- Build a reliable classifier to detect phishing URLs from URL-derived features.
- Provide an easy-to-use API and UI to test URLs and demonstrate the model.
- Make the system reproducible and deployable (local or Docker).


## 3. Dataset
- File: `data/Website Phishing.csv`
- Typical contents: URL records with labels and engineered features used for training (for example: `having_ip_address`, `url_length`, `sslfinal_state`, `sfh`, `popupwidnow`, `request_url`, `url_of_anchor`, `web_traffic`, `age_of_domain`, ...).
- Notes: Verify class balance and any preprocessing steps in `models/model_training.py` and the notebook.


## 4. Feature Engineering
The project currently uses URL-derived features. Common feature types included (and their meanings):
- `having_ip_address` — 1 if an IP address is present in the URL, -1 otherwise
- `url_length` — categorised URL length (1 = short/likely safe, 0 = suspicious, -1 = very long)
- `sslfinal_state` — 1 if HTTPS present, -1 otherwise
- `sfh` — server form handler feature (sign of form action pointing to suspicious domain)
- `popupwidnow` — indicates presence of popup windows
- `request_url` — external resource requests proportion
- `url_of_anchor` — proportion of anchors linking to external domains
- `web_traffic` — relative web traffic indicator
- `age_of_domain` — domain age indicator

Note: Exact calculation rules and thresholds live in `models/model_training.py` and the feature extraction logic in `app/appUI.py` (for live inference). Improve feature extraction for production by adding checks for suspicious keywords, subdomain anomalies, TLD reputation, '@' symbol, excessive redirect chains, and WHOIS-based domain age checks.


## 5. Model Selection & Training
- Model used: Random Forest Classifier
- Training script: `models/model_training.py` (script) and `models/model_training_notebook.ipynb` (exploratory notebook)
- Saved model: `models/RFC_best_model.pkl` (loaded by `app/app.py` for inference)

Reproducibility steps (high level):
1. Prepare data in `data/Website Phishing.csv`.
2. Run preprocessing and feature extraction in `models/model_training.py` or the notebook.
3. Train multiple candidate classifiers and select best using cross-validation.
4. Save the final model to `models/RFC_best_model.pkl` using `pickle`.

Because exact train/test splits and hyperparameters are project-specific, consult the notebook for detailed commands and tuning results.


## 6. Evaluation
Metrics to compute (recommended):
- Accuracy
- Precision, Recall, F1-score (important due to class imbalance and cost sensitivity)
- ROC-AUC
- Confusion matrix

Where to compute metrics:
- The training notebook (`models/model_training_notebook.ipynb`) should contain evaluation runs. If the notebook does not include these, add a metrics section that computes the above on a held-out test set.

Note: This report intentionally does not fabricate numeric results. Please run the training notebook to record exact evaluation figures and paste them into the "Results" section below.


## 7. Results (How to populate)
- After running the training notebook or script, fill in:
  - Final model hyperparameters
  - Train/validation/test scores
  - Confusion matrix and ROC curve images (save them under `reports/` or `docs/` if desired)

Example placeholder layout (replace with real numbers):
- Accuracy: <replace-with-value>
- Precision (Phishing class): <replace>
- Recall (Phishing class): <replace>
- F1-score (Phishing class): <replace>
- ROC-AUC: <replace>


## 8. API Design
- File: `app/app.py`
- Endpoints:
  - `GET /` — simple welcome message
  - `GET /health` — health check (`{"status": "healthy"}`)
  - `POST /predict/` — model inference endpoint
    - Request JSON body: example fields (all integers):
      ```json
      {
        "sfh": -1,
        "popupwidnow": 0,
        "sslfinal_state": -1,
        "request_url": -1,
        "url_of_anchor": -1,
        "web_traffic": 0,
        "url_length": -1,
        "age_of_domain": -1,
        "having_ip_address": 1
      }
      ```
    - Response JSON:
      ```json
      {
        "prediction": 0,
        "prediction_text": "Phishing",
        "probability": 0.9512
      }
      ```

Notes on behavior: `app/app.py` currently loads the model from `models/RFC_best_model.pkl` at import time. Ensure the `models` path is accessible when running the API.


## 9. Front-end (Streamlit UI)
- File: `app/appUI.py`
- Provides an interactive page for entering a URL, extracting features client-side, sending them to the API, and showing results with animations.
- The client-side feature extraction is intentionally simplified for demo purposes. To match model training precisely, ensure the same feature extraction pipeline is used in both training (dataset preprocessing) and inference (UI + API). Differences here often cause mismatches (e.g. always predicting "Legitimate").

Common mismatch sources:
- Different encoding/thresholds of categorical features
- Missing features expected by the model
- Training-time feature scaling/ordering not applied in inference


## 10. Troubleshooting & Common Issues
1. Always ensure the API is running on `http://localhost:8000` before using the Streamlit UI.
2. If all predictions are "Legitimate":
   - Confirm UI feature extraction outputs realistic feature values for suspicious URLs.
   - Print the `df` sent to the model in `app/app.py` (temporary logging) to verify shape and column ordering.
   - Confirm the model expects the same feature order and names used by the UI.
3. Serialization errors:
   - If `models/RFC_best_model.pkl` fails to load, retrain the model using the same scikit-learn version as in your environment.


## 11. How to Run (Local)
1. Create and activate virtual environment (Windows PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # or venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the API (option A: run via script):

```powershell
python app/app.py
```

or (option B: use uvicorn if preferred):

```powershell
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

3. Start the Streamlit UI:

```powershell
streamlit run app/appUI.py
```

4. Open the UI in your browser (Streamlit will show the local URL) and/or use `http://localhost:8000/docs` to explore the API.


## 12. Docker (Optional)
Example `Dockerfile` (recommended improvements: use a non-root user, pin dependencies):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app/app.py"]
```

Build and run:

```powershell
docker build -t phishing-api .
docker run -p 8000:8000 phishing-api
```


## 13. Security and Privacy Considerations
- Do not send sensitive or private URLs to third-party services during debugging.
- If deploying publicly, lock down the API with authentication and HTTPS.
- Sanitize and rate-limit inputs to avoid abuse.


## 14. Future Work and Improvements
- Improve feature extraction to capture phishing signals (keyword checks, subdomain pattern analysis, TLD reputation, WHOIS checks).
- Retrain model with cross-validation and hyperparameter tuning; log experiments with MLflow or Weights & Biases.
- Add more robust input validation and a consistent inference pipeline shared by training and serving.
- Add CI tests for API and UI, and unit tests for feature extraction functions.
- Add a model monitoring dashboard to watch for concept drift.


## 15. Appendix

### 15.1 Sample Test URLs
Legitimate examples:
- https://www.google.com
- https://www.microsoft.com
- https://www.wikipedia.org

Suspicious / phishing examples:
- http://192.168.1.1/login
- http://secure-login.com.fake-site.ru
- http://paypal.com.account-security-alert.com
- http://free-gift-card.win


### 15.2 Example API Request (curl)

```powershell
curl -X POST http://localhost:8000/predict/ -H "Content-Type: application/json" -d "{
  \"sfh\": -1,
  \"popupwidnow\": 0,
  \"sslfinal_state\": -1,
  \"request_url\": -1,
  \"url_of_anchor\": -1,
  \"web_traffic\": 0,
  \"url_length\": -1,
  \"age_of_domain\": -1,
  \"having_ip_address\": 1
}"
```


### 15.3 Reproducibility checklist
- [ ] Confirm Python version and packages in `requirements.txt`
- [ ] Run `models/model_training.py` or the notebook to reproduce the model
- [ ] Evaluate on a held-out test set and save results under `reports/`


## 16. Contact & Maintainer
- Maintainer: project author (see repository)
- For questions or contributions, open an issue or a pull request.


---

This report is generated to help document the project, reproduce the experiments, and guide future improvements. Fill in the evaluation numbers and experiment logs after re-running the training scripts or notebook so the Results section contains precise performance metrics.

---

## 17. Algorithms and Implementation Details

### 17.1 Algorithms Used
- Random Forest Classifier (scikit-learn)
  - Ensemble of decision trees using bootstrap aggregation (bagging).
  - Each tree is trained on a random sample of the data and a random subset of features.
  - Final prediction is by majority vote (classification) or average (regression).
  - Strengths: robust to overfitting vs single decision trees, handles heterogeneous features, provides feature importance.
  - Hyperparameters to tune: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`.

- (Optional) Logistic Regression
  - Baseline linear classifier providing calibrated probabilities.
  - Useful as a simple and interpretable baseline.

- (Optional) Gradient Boosting (XGBoost / LightGBM)
  - Boosted tree models offering state-of-the-art performance in many tabular tasks.
  - More sensitive to hyperparameter tuning and can require more compute.

### 17.2 Feature Pipeline (Implementation Notes)
- Data ingestion: `pandas` used to load CSV dataset.
- Preprocessing steps typically include:
  - Cleaning malformed URLs and normalizing text
  - Parsing URL components using `urllib.parse` (domain, path, query, subdomain)
  - Creating binary/categorical features (presence of IP, HTTPS, '@' symbol) and numeric bins (URL length categories)
  - Encoding categorical features (if any) and ensuring consistent column order
  - Train/test split with stratification on the target label to preserve class balance

### 17.3 Model Persistence and Serving
- The trained scikit-learn model is serialized with `pickle` to `models/RFC_best_model.pkl`.
- `app/app.py` loads this model at import time and exposes `POST /predict/` for inference.

### 17.4 Why Legitimate Prediction Happens Often
- If feature extraction in the UI differs from training preprocessing (order, scaling, mapping of categorical labels), model inputs will not match and predictions will be unreliable.
- Verify column names and ordering, and ensure any encoding/scaling pipeline applied at training time is also applied during serving (recommended: use scikit-learn `Pipeline` and persist it).

---

## 18. Packages and Dependency Details

Below are the project dependencies from `requirements.txt` with recommended version pins. Pinning ensures reproducibility across environments.

Recommended minimal `requirements.txt` (example pins):

```text
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.3.2
pickle-mixin==1.0.2
fastapi==0.101.0
uvicorn==0.23.0
pydantic==2.5.2
streamlit==1.30.0
requests==2.31.0
matplotlib==3.8.1
seaborn==0.12.2
streamlit-lottie==0.0.4
```

Notes:
- `pickle-mixin` is rarely needed; Python's built-in `pickle` is typically sufficient. Keep only if your training pipeline requires it.
- `streamlit-lottie` is used by `app/appUI.py` for the Lottie animations introduced earlier—install it if you use the enhanced UI.

### 18.1 How to create a reproducible environment
1. Create virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install pinned dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Freeze the environment to `requirements-lock.txt` after installation:

```powershell
pip freeze > requirements-lock.txt
```

---

## 19. Expanded Implementation Checklist

- [ ] Verify dataset integrity and class balance in `data/Website Phishing.csv`
- [ ] Implement a scikit-learn `Pipeline` that includes preprocessing (feature transforms) and the classifier; save the entire pipeline with `joblib` or `pickle`.
- [ ] Ensure `app/app.py` loads the saved pipeline and applies it to incoming JSON in the same way as training.
- [ ] Add logging in `app/app.py` to log incoming requests and model inputs (avoid sensitive data in logs).
- [ ] Improve `app/appUI.py` feature extraction to match the pipeline exactly or move extraction server-side.
- [ ] Add unit tests for feature extraction and API endpoints.
- [ ] Add model evaluation scripts that output performance metrics and save plots to `reports/`.

---

## 20. Sources and References
- Scikit-learn documentation: https://scikit-learn.org/stable/
- FastAPI documentation: https://fastapi.tiangolo.com/
- Streamlit documentation: https://streamlit.io/
- Feature ideas for phishing detection: academic papers and blog posts on URL-based phishing detection algorithms.


# Phishing Detection ML API

## Overview
This project provides a machine learning-powered API for detecting phishing websites using FastAPI, Docker, and a trained Random Forest model. It predicts whether a website is legitimate or phishing based on various URL features.

---

## Features
- **FastAPI** backend for fast, interactive REST API
- **Machine Learning** model (Random Forest) for phishing detection
- **Docker** support for easy deployment
- **Health check** endpoint
- **Interactive API docs** via Swagger UI (`/docs`)

---

## Project Structure
```
Phishing_Detection_ML_FastAPI/
│
├── app/
│   ├── app.py                # FastAPI application (main API)
│   ├── appUI.py              # (Optional) UI code for frontend
│   ├── test_api.py           # API test scripts
│   └── __pycache__/          # Python cache files
│
├── data/
│   └── Website Phishing.csv  # Dataset for training/testing
│
├── models/
│   ├── model_training.py     # Model training script
│   ├── model_training_notebook.ipynb # Jupyter notebook for training
│   └── RFC_best_model.pkl    # Trained Random Forest model
│
└── requirements.txt          # Python dependencies
```

---

## Setup & Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/selvin-paul-raj/Phishing_Detection_ML_FastAPI
cd Phishing_Detection_ML_FastAPI
```

### 2. Install Dependencies
Create a virtual environment (recommended):
```powershell
python -m venv venv
venv\Scripts\activate
```
Install required packages:
```powershell
pip install -r requirements.txt
```

3. **Run the Application**
   ```bash
   # Start the FastAPI backend (in one terminal)
   uvicorn app.app:app --host 0.0.0.0 --port 8000
   
   # Start the Streamlit frontend (in another terminal)
   streamlit run app/appUI.py
   ```

4. **Access the Application**
   - Streamlit UI: http://localhost:8501

### 5. Docker Deployment (Optional)
To run with Docker:
1. Create a `Dockerfile` (see below for example)
2. Build and run the container:
   ```powershell
   docker build -t phishing-api .
   docker run -p 8000:8000 phishing-api
   ```

---

## API Endpoints

### `GET /`
- Welcome message.

### `GET /health`
- Health check endpoint.
- Returns `{ "status": "healthy" }`

### `POST /predict/`
- Predicts if a website is phishing or legitimate.
- **Request Body (JSON):**
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
- **Response:**
  ```json
  {
    "prediction": 0,
    "prediction_text": "Phishing",
    "probability": 0.95
  }
  ```

---

## Example Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app/app.py"]
```

---

## Testing
- Use `app/test_api.py` for API tests.
- You can also use Swagger UI (`/docs`) for manual testing.

---

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-xyz`)
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or support, contact the maintainer.

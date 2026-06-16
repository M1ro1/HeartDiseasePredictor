# Heart Disease Prediction

A full-stack heart disease risk prediction project that combines machine learning, explainable AI, and a modern web interface.

## Overview

This application estimates the risk of heart disease from 11 clinical indicators and provides:

- a binary prediction result,
- a probability score,
- SHAP-based model explanations,
- downloadable PDF reports,
- prediction history for authenticated users.

The project includes three main parts:

- `main.py` — machine learning training and evaluation pipeline
- `backend/` — FastAPI backend, authentication, and database integration
- `frontend/` — React frontend for entering patient data and viewing results

## Features

- Heart disease risk prediction from clinical data
- Model training and evaluation with multiple algorithms
- Explainable AI using SHAP
- PDF report generation
- User registration and login
- Saved prediction history
- Docker support for full-stack deployment

## Tech Stack

### Backend
- Python
- FastAPI
- SQLAlchemy
- Alembic
- PostgreSQL

### Machine Learning
- scikit-learn
- XGBoost
- SHAP
- pandas
- NumPy
- Joblib

### Frontend
- React
- Vite
- React Router
- Tailwind CSS

## Project Structure

```text
HeartDiseasePrediction/
├── main.py
├── backend/
│   ├── app/
│   │   ├── api.py
│   │   ├── db/
│   │   └── ml/
│   ├── migrations/
│   └── tests/
├── frontend/
│   └── src/
├── misc/
│   ├── heart_disease_uci.csv
│   ├── random_forest_model.joblib
│   └── preprocessor.joblib
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Clinical Inputs

The model uses the following indicators:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression (`oldpeak`)
- ST slope

## How It Works

1. The training pipeline loads and prepares the dataset.
2. The best model and preprocessing pipeline are saved to `misc/`.
3. The FastAPI backend loads the saved artifacts.
4. The frontend sends patient data to the backend.
5. The backend returns a prediction, probability, SHAP explanation, and PDF report.

## API Endpoints

- `GET /` — redirect to the prediction page
- `POST /registration` — register a new user
- `POST /token` — log in and receive an access token
- `POST /predict` — generate a heart disease prediction
- `GET /history` — retrieve saved analysis history

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Make sure the required values are available in `.env`.

Example:

```env
DB_NAME=heart_disease_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=db
DB_PORT=5432
SECRET_KEY=your-secret-key
VITE_API_URL=http://localhost:8000
```

## Running Locally

### Train the model

```bash
python main.py
```

### Start the backend

```bash
uvicorn backend.app.api:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

## Running with Docker

```bash
docker compose up --build
```

This starts:

- PostgreSQL
- FastAPI backend
- React frontend

## Notes

- The backend expects the trained model and preprocessor files in `misc/`.
- Prediction history is available for authenticated users.
- SHAP visualizations and PDF reports are generated for each prediction.

## Disclaimer

This project is intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.


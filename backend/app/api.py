import threading
from contextlib import asynccontextmanager

from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi import FastAPI,Depends, HTTPException, status

import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import os
import asyncio

from .ml.generate_pdf_file import PatientDataFile
from dotenv import load_dotenv

from .db.schemas import UserOut, UserCreate
from .db.crud import create_user, get_user_by_username
from .db.database import get_db

from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'misc', 'random_forest_model.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'misc', 'preprocessor.joblib')

plt_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Model or preprocessor not found in {os.path.dirname(MODEL_PATH)}")

    app.state.model = joblib.load(MODEL_PATH)
    app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)
    app.state.explainer = shap.TreeExplainer(app.state.model)

    yield

    app.state.model = None
    app.state.preprocessor = None
    app.state.explainer = None

app = FastAPI(title="Heart Disease API", lifespan=lifespan)

class PatientData(BaseModel):
    age:int
    sex:str
    cp:str
    trestbps:float
    chol:float
    fbs:str
    restecg:str
    thalch:float
    exang:str
    oldpeak:float
    slope:str


def generate_shap_plot(explainer, processed_df):
    with plt_lock:
        shap_explanation = explainer(processed_df)
        shap_class1 = shap_explanation[0, :, 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_class1, max_display=10, show=False)
        plt.title("Impact of indicators on increasing/decreasing risk", fontsize=12, pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)

        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

@app.get("/")
def home_page():
    return RedirectResponse(url='/predict')

@app.post("/registration", response_model=UserOut)
async def register_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):

    existing_user = await get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this username already exists"
        )

    new_user = await create_user(db, user_data)

    return new_user

@app.post("/predict")
async def predict(data: PatientData):
    model = app.state.model
    preprocessor = app.state.preprocessor
    explainer = app.state.explainer

    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    preprocessed_input = preprocessor.transform(input_df)

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in raw_feature_names]

    processed_df = pd.DataFrame(preprocessed_input, columns=feature_names)

    prediction = int(model.predict(processed_df)[0])
    probability = float(model.predict_proba(processed_df)[0][1] * 100)

    img_base64 = await asyncio.to_thread(generate_shap_plot, explainer, processed_df)
    generated_file = await asyncio.to_thread(
        PatientDataFile.generate_pdf, input_dict, probability, img_base64
    )

    pdf_base64 = base64.b64encode(generated_file).decode('utf-8')

    return {
        "prediction": prediction,
        "probability": probability,
        'shap_plot': img_base64,
        'patient_data_file': pdf_base64
    }


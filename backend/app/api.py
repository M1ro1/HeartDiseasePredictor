import threading
from contextlib import asynccontextmanager

from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi import FastAPI,Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import os
import asyncio
import uuid
from typing import Optional

from .ml.generate_pdf_file import PatientDataFile
from dotenv import load_dotenv

from .db.schemas import UserOut, UserCreate, HistoryRead
from .db.crud import create_user, get_user_by_username, login_user, get_current_user
from .db.database import get_db
from .db.models import UserTable, AnalysisHistory

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

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

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(),
                db: AsyncSession = Depends(get_db)):

    user = await login_user(db, form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    session_token = str(uuid.uuid4())

    user.session_token = session_token

    await db.commit()
    await db.refresh(user)

    return {
        "access_token": session_token,
        "token_type": "bearer",
        "username": user.username
    }
@app.post("/predict")
async def predict(data: PatientData,
                  db: AsyncSession = Depends(get_db),
                  user: Optional[UserTable] = Depends(get_current_user)
                  ):

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

    if user:
        new_history = AnalysisHistory(
            user_id=user.id,
            input_data=data.dict(),
            probability=probability,
            prediction=prediction,
            shap_image_base64=img_base64
        )
        db.add(new_history)
        await db.commit()

    return {
        "prediction": prediction,
        "probability": probability,
        'shap_plot': img_base64,
        'patient_data_file': pdf_base64
    }

@app.get("/history", response_model=list[HistoryRead])
async def get_history(
        db: AsyncSession = Depends(get_db),
        user: UserTable = Depends(get_current_user)
):

    if not user:
        raise HTTPException(status_code=401, detail="Not authorized")

    query = (
        select(AnalysisHistory)
        .where(AnalysisHistory.user_id == user.id)
        .order_by(AnalysisHistory.created_at.desc())
    )

    result = await db.execute(query)
    history_list = result.scalars().all()

    return history_list
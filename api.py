from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import os

matplotlib.use('Agg')

app = FastAPI(title="Heart Disease API")

MODEL_PATH = os.path.join('./misc/', 'random_forest_model.joblib')
PREPROCESSOR_PATH = os.path.join('./misc/', 'preprocessor.joblib')

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

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


@app.post("/predict")
def predict(data: PatientData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")

    input = data.dict()
    input_df = pd.DataFrame([input])

    preprocessed_input = preprocessor.transform(input_df)

    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    feature_names = num_features + list(
    preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features))

    processed_df = pd.DataFrame(preprocessed_input, columns=feature_names)
    prediction = int(model.predict(processed_df)[0])
    probability = float(model.predict_proba(processed_df)[0][1] * 100)

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(processed_df)
    shap_class1 = shap_explanation[0, :, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_class1, max_display=10, show=False)
    plt.title("Impact of indicators on increasing/decreasing risk", fontsize=12, pad=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return {
        "prediction": prediction,
        "probability": probability,
        'shap_plot': img_base64}


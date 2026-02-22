import matplotlib.pyplot as plt
import numpy as np

import shap


def explain_model_shap(rf_model, X_test_df):
    explainer = shap.TreeExplainer(rf_model)

    shap_explanation = explainer(X_test_df)

    shap_class1 = shap_explanation[:, :, 1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_class1, X_test_df, show=False)
    plt.title("Global importance of features (SHAP Summary)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

    probs = rf_model.predict_proba(X_test_df)[:, 1]
    patient_idx = np.argmax(probs)

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_class1[patient_idx], max_display=10, show=False)

    patient_risk = probs[patient_idx] * 100
    plt.title(f"Forecast Explanation (Risk: {patient_risk:.2f}%)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

    return shap_explanation


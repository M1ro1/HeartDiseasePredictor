import os
import numpy as np
from collections import Counter
from dython.nominal import associations
from mlxtend.evaluate import mcnemar_table, mcnemar
import joblib
from backend.data import load_dataset, prepare_X_y, split_data
from backend.preprocessing import build_preprocessors, apply_preprocessors
from backend.models import tune_models
from backend.explainability import explain_model_shap
from backend.utils import save_full_dataset


DATA_PATH = os.path.join("./misc/", "heart_disease_uci.csv")


def main():
    df = load_dataset(DATA_PATH)

    X, y = prepare_X_y(df)

    print(f"Початкова форма X (після завантаження): {df.shape}")
    print(f"Форма X після підготовки: {X.shape}")
    print(f"Форма y після синхронізації: {y.shape}")

    print("\nПобудова матриці асоціацій (Mixed Data Types)")
    df_analysis = X.copy()
    df_analysis['target'] = y

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'target']
    results = associations(
        df_analysis,
        nominal_columns=categorical_cols,
        figsize=(15, 12),
        cmap='coolwarm',
        title="Матриця асоціацій факторів ризику серцевих захворювань",
        display_rows='all',
        display_columns='all',
    )

    corr_matrix = results['corr']
    print("\nТоп-5 факторів, що найбільше асоціюються з target:")
    print(corr_matrix['target'].drop('target').abs().sort_values(ascending=False).head(5))

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"\n--- Розміри вибірок ---")
    print(f"Навчальна (X_train):   {X_train.shape} | Баланс (y_train): {Counter(y_train)}")
    print(f"Валідаційна (X_val): {X_val.shape} | Баланс (y_val): {Counter(y_val)}")
    print(f"Тестова (X_test):     {X_test.shape} | Баланс (y_test): {Counter(y_test)}")

    numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

    preprocessor, viz_preprocessor = build_preprocessors(numeric_features, categorical_features)

    (
        X_train_df,
        X_val_df,
        X_test_df,
        X_train_viz_df,
        X_val_viz_df,
        X_test_viz_df,
    ) = apply_preprocessors(preprocessor, viz_preprocessor, X_train, X_val, X_test, numeric_features, categorical_features)

    print("\nОбробка завершена")
    print(f"Кількість ознак до обробки: {X_train.shape[1]}")
    print(f"Кількість ознак після обробки (з OneHot): {X_train_df.shape[1]}")

    print("\nПерші 5 рядків оброблених навчальних даних (X_train_df):")
    print(X_train_df.head().to_string())

    print("\nПеревірка на наявність NaN у X_train (обробленому):")
    print(f"Залишилось NaN: {X_train_df.isnull().sum().sum()}")

    out_path = os.path.join("./misc/", "full_dataset.csv")
    save_full_dataset(X_train_df.assign(target=y_train), X_val_df.assign(target=y_val), X_test_df.assign(target=y_test), out_path)

    best_models = tune_models(X_train_df, X_val_df, y_train, y_val)

    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix

    print("\nФінальні результати на TEST set (15%)")
    print(f"{'Модель':<10} | {'Accuracy':<10} | {'Recall':<10} | {'AUC-ROC':<10}")

    test_predictions = {}
    results_metrics = {}

    for name, model in best_models.items():
        y_pred = model.predict(X_test_df)
        y_prob = model.predict_proba(X_test_df)[:, 1]

        test_predictions[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results_metrics[name] = {'Accuracy': acc, 'Recall': rec, 'AUC': auc}
        print(f"{name:<10} | {acc:.4f} | {rec:.4f} | {auc:.4f}")

    print(" ЕТАП 3: АНАЛІЗ TRAIN vs TEST ТА МЕТРИКИ ")
    print(f"{'Модель':<6} | {'Data':<5} | {'Accuracy':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<8} | {'AUC':<8}")

    for name, model in best_models.items():
        y_train_pred = model.predict(X_train_df)
        y_train_prob = model.predict_proba(X_train_df)[:, 1]

        train_acc = accuracy_score(y_train, y_train_pred)
        train_rec = recall_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_prob)

        y_test_pred = model.predict(X_test_df)
        y_test_prob = model.predict_proba(X_test_df)[:, 1]

        test_predictions[name] = y_test_pred

        test_acc = accuracy_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)

        print(f"{name:<6} | Train | {train_acc:.4f} | {train_rec:.4f} | {train_prec:.4f} | {train_f1:.4f} | {train_auc:.4f}")
        print(f"{'':<6} | Test | {test_acc:.4f} | {test_rec:.4f} | {test_prec:.4f} | {test_f1:.4f} | {test_auc:.4f}")

    rf_model = best_models.get('RF')
    if rf_model is not None:
        explain_model_shap(rf_model, X_test_df)

    print("Confusion Matrix")
    print("Format:\n        ]")
    print("FN (False Negative) - це пропущені хворі (найгірша помилка).")

    for name, y_pred in test_predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name}")
        print(cm)
        print(f"Пропущено хворих (FN): {fn}")
        print(f"Хибна тривога (FP): {fp}")

    def run_hypothesis_test(y_true, pred_dict, model1_name, model2_name, alpha=0.05):
        print(f"\nПеревірка гіпотез: {model1_name} vs {model2_name}")

        y_pred1 = pred_dict[model1_name]
        y_pred2 = pred_dict[model2_name]

        tb = mcnemar_table(
            y_target=np.array(y_true),
            y_model1=np.array(y_pred1),
            y_model2=np.array(y_pred2),
        )

        print("Таблиця спряженості (де моделі помиляються по-різному):")
        print(tb)

        chi2, p = mcnemar(ary=tb, corrected=True)

        print(f"P-value: {p:.4f}")
        if p < alpha:
            print("ВІДХИЛЯЄМО H0. Різниця є СТАТИСТИЧНО ЗНАЧУЩОЮ.")
            print("Моделі працюють по-різному на цій тестовій вибірці.")
        else:
            print("НЕ ВІДХИЛЯЄМО H0. Різниця НЕ є статистично значущою.")
            print("Різниця в точності може бути випадковою.")

    print(" ЕТАП 3: СТАТИСТИЧНЕ ПОРІВНЯННЯ МОДЕЛЕЙ ")

    try:
        run_hypothesis_test(y_test, test_predictions, 'RF', 'LR')
        run_hypothesis_test(y_test, test_predictions, 'XGB', 'LR')
    except Exception:
        print("Помилка при запуску тесту макнемара — можливо відсутні передані прогнози для деяких моделей")

    if rf_model is not None:
        joblib.dump(rf_model, os.path.join(os.getcwd(), 'random_forest_model.joblib'))
        joblib.dump(preprocessor, os.path.join(os.getcwd(), 'preprocessor.joblib'))


if __name__ == '__main__':
    main()

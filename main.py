import os
import numpy as np
from collections import Counter
from dython.nominal import associations
from mlxtend.evaluate import mcnemar_table, mcnemar
import joblib
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix

from backend.data import load_dataset, prepare_X_y, split_data
from backend.preprocessing import build_preprocessors, apply_preprocessors
from backend.models import tune_models
from backend.explainability import explain_model_shap
from backend.utils import save_full_dataset

from dotenv import load_dotenv

load_dotenv()

class HeartDiseasePipeline:
    def __init__(self, data_path: str, output_dir: str = './misc/' ):
        self.data_path = data_path
        self.output_dir = output_dir

        self.df = None
        self.X = None
        self.y = None
        self.best_models = {}
        self.test_predictions = {}
        self.preprocessor = None

        self.numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

    def load_and_split_data(self):
        self.df = load_dataset(self.data_path)
        self.X, self.y = prepare_X_y(self.df)

        print(f"Початкова форма X (після завантаження): {self.df.shape}")
        print(f"Форма X після підготовки: {self.X.shape}")
        print(f"Форма y після синхронізації: {self.y.shape}")

        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test) = split_data(self.X, self.y)

        print("\n--- Розміри вибірок ---")
        print(f"Навчальна: {self.X_train.shape} | Баланс: {Counter(self.y_train)}")
        print(f"Валідаційна: {self.X_val.shape} | Баланс: {Counter(self.y_val)}")
        print(f"Тестова: {self.X_test.shape} | Баланс: {Counter(self.y_test)}")

    def run_eda(self):
        print("\nПобудова матриці асоціацій (Mixed Data Types)")
        df_analysis = self.X.copy()
        df_analysis['target'] = self.y

        categorical_cols = self.categorical_features + ['target']

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

    def preprocess_data(self):
        self.preprocessor, viz_preprocessor = build_preprocessors(self.numeric_features, self.categorical_features)

        (self.X_train_df, self.X_val_df, self.X_test_df, _, _, _) = apply_preprocessors(
            self.preprocessor, viz_preprocessor,
            self.X_train, self.X_val, self.X_test,
            self.numeric_features, self.categorical_features
        )

        print(f"Кількість ознак до обробки: {self.X_train.shape[1]}")
        print(f"Кількість ознак після обробки (з OneHot): {self.X_train_df.shape[1]}")

        print("\nПерші 5 рядків оброблених навчальних даних (X_train_df):")
        print(self.X_train_df.head().to_string())

        print("\nПеревірка на наявність NaN у X_train (обробленому):")
        print(f"Залишилось NaN: {self.X_train_df.isnull().sum().sum()}")

        out_path = os.path.join(self.output_dir, "full_dataset.csv")

        save_full_dataset(
            self.X_train_df.assign(target=self.y_train),
            self.X_val_df.assign(target=self.y_val),
            self.X_test_df.assign(target=self.y_test),
            out_path
        )

    def train_models(self):
        self.best_models = tune_models(self.X_train_df, self.X_val_df, self.y_train, self.y_val)

    def evaluate_models(self):
        print("\n ЕТАП 3: АНАЛІЗ TRAIN vs TEST ТА МЕТРИКИ ")
        print(f"{'Модель':<6} | {'Data':<5} | {'Acc':<6} | {'Recall':<6} | {'Prec':<6} | {'F1':<6} | {'AUC':<6}")

        for name, model in self.best_models.items():
            y_train_pred = model.predict(self.X_train_df)
            y_train_prob = model.predict_proba(self.X_train_df)[:, 1]

            print(f"{name:<6} | Train | {accuracy_score(self.y_train, y_train_pred):.4f} | "
                  f"{recall_score(self.y_train, y_train_pred):.4f} | {precision_score(self.y_train, y_train_pred):.4f} | "
                  f"{f1_score(self.y_train, y_train_pred):.4f} | {roc_auc_score(self.y_train, y_train_prob):.4f}")

            y_test_pred = model.predict(self.X_test_df)
            y_test_prob = model.predict_proba(self.X_test_df)[:, 1]
            self.test_predictions[name] = y_test_pred

            print(f"{'':<6} | Test  | {accuracy_score(self.y_test, y_test_pred):.4f} | "
                  f"{recall_score(self.y_test, y_test_pred):.4f} | {precision_score(self.y_test, y_test_pred):.4f} | "
                  f"{f1_score(self.y_test, y_test_pred):.4f} | {roc_auc_score(self.y_test, y_test_prob):.4f}")

            cm = confusion_matrix(self.y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            print(f"--- {name} Confusion Matrix ---")
            print(f"Пропущено хворих (FN): {fn} | Хибна тривога (FP): {fp}\n")

    def run_statistical_tests(self):
        print("\n ЕТАП 3: СТАТИСТИЧНЕ ПОРІВНЯННЯ МОДЕЛЕЙ ")

        def test_pair(model1, model2):
            if model1 in self.test_predictions and model2 in self.test_predictions:
                tb = mcnemar_table(
                    y_target=np.array(self.y_test),
                    y_model1=np.array(self.test_predictions[model1]),
                    y_model2=np.array(self.test_predictions[model2]),
                )
                chi2, p = mcnemar(ary=tb, corrected=True)
                print(f"{model1} vs {model2} -> P-value: {p:.4f} "
                      f"({'ЗНАЧУЩА' if p < 0.05 else 'НЕ значуща'} різниця)")

        try:
            test_pair('RF', 'LR')
            test_pair('XGB', 'LR')
        except Exception as e:
            print(f"Помилка при запуску тесту Макнемара: {e}")

    def run_explainability_and_save(self):
        rf_model = self.best_models.get('RF')
        if rf_model is not None:
            explain_model_shap(rf_model, self.X_test_df)

            joblib.dump(rf_model, os.path.join(self.output_dir, 'random_forest_model.joblib'))
            joblib.dump(self.preprocessor, os.path.join(self.output_dir, 'preprocessor.joblib'))
            print("\nМодель та препроцесор успішно збережено.")

    def execute(self):
        self.load_and_split_data()
        self.run_eda()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        self.run_statistical_tests()
        self.run_explainability_and_save()


if __name__ == '__main__':
    DATA_PATH = os.path.join("./misc/", "heart_disease_uci.csv")
    pipeline = HeartDiseasePipeline(data_path = DATA_PATH)
    pipeline.execute()

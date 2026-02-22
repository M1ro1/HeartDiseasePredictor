from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def tune_models(X_train_df, X_val_df, y_train, y_val):
    X_combined = __import__('pandas').concat([X_train_df, X_val_df], axis=0)
    y_combined = __import__('pandas').concat([y_train, y_val], axis=0)

    split_index = [-1] * len(X_train_df) + [0] * len(X_val_df)
    ps = PredefinedSplit(test_fold=split_index)

    best_models = {}

    lr_params = {'C': [0.01, 0.1, 1, 10, 20, 100]}
    grid_lr = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42), lr_params, cv=ps, scoring='recall'
    )
    grid_lr.fit(X_combined, y_combined)
    best_models['LR'] = grid_lr.best_estimator_

    rf_params = {
        'n_estimators': [100, 200, 300, 500, 700, 1000],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 3, 4, 5, 10, 20],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42), rf_params, cv=ps, scoring='recall', n_jobs=-1
    )
    grid_rf.fit(X_combined, y_combined)
    best_models['RF'] = grid_rf.best_estimator_

    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [2, 4, 5],
    }
    grid_xgb = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        xgb_params,
        cv=ps,
        scoring='recall',
        n_jobs=-1,
    )
    grid_xgb.fit(X_combined, y_combined)
    best_models['XGB'] = grid_xgb.best_estimator_

    return best_models


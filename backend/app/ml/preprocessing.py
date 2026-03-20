import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessors(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='passthrough',
    )

    viz_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    viz_categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    viz_preprocessor = ColumnTransformer(
        transformers=[
            ('num', viz_numeric_transformer, numeric_features),
            ('cat', viz_categorical_transformer, categorical_features),
        ],
        verbose_feature_names_out=False,
    )

    return preprocessor, viz_preprocessor


def apply_preprocessors(preprocessor, viz_preprocessor, X_train, X_val, X_test, numeric_features, categorical_features):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    processed_feature_names = (
        numeric_features
        + list(
            preprocessor.named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(categorical_features)
        )
    )

    X_train_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_processed, columns=processed_feature_names, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)

    X_train_viz = viz_preprocessor.fit_transform(X_train)
    X_val_viz = viz_preprocessor.transform(X_val)
    X_test_viz = viz_preprocessor.transform(X_test)

    cols_viz = numeric_features + categorical_features
    X_train_viz_df = pd.DataFrame(X_train_viz, columns=cols_viz)
    X_val_viz_df = pd.DataFrame(X_val_viz, columns=cols_viz)
    X_test_viz_df = pd.DataFrame(X_test_viz, columns=cols_viz)

    for col in numeric_features:
        X_train_viz_df[col] = X_train_viz_df[col].astype(float)
        X_val_viz_df[col] = X_val_viz_df[col].astype(float)
        X_test_viz_df[col] = X_test_viz_df[col].astype(float)

    return (
        X_train_df,
        X_val_df,
        X_test_df,
        X_train_viz_df,
        X_val_viz_df,
        X_test_viz_df,
    )


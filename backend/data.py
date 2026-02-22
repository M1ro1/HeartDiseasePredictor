import pandas as pd
import numpy as np


def load_dataset(path: str):
    df = pd.read_csv(path)
    return df


def prepare_X_y(df: pd.DataFrame):
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop(columns=['num', 'id', 'dataset'])
    X = X.drop(columns=[c for c in ['ca', 'thal'] if c in X.columns])

    if 'trestbps' in X.columns:
        X['trestbps'] = X['trestbps'].replace(0, np.nan)
    if 'chol' in X.columns:
        X['chol'] = X['chol'].replace(0, np.nan)

    if 'restecg' in X.columns:
        X = X.dropna(subset=['restecg'])
        y = y.loc[X.index]

    return X, y


def split_data(X, y, test_size=0.15, val_share_of_rest=0.17647058823529413, random_state=43):
    from sklearn.model_selection import train_test_split

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_share_of_rest,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

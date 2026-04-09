import pandas as pd
import numpy as np
from app.ml.data import prepare_X_y, split_data

def test_prepare_X_y():
    df = pd.DataFrame({
        'id': [1,2,3],
        'dataset': ['train','test','train'],
        'num': [0, 2, 0],
        'ca': [1, 0, 2],
        'trestbps': [0, 120, 130],
        'restecg': [1, np.nan, 0]
    })

    X,y = prepare_X_y(df)

    assert 'num' not in X.columns
    assert 'id' not in X.columns
    assert 'ca' not in X.columns

    assert len(X) == 2
    assert len(y) == 2

    assert y.loc[0] == 0
    assert y.loc[2] == 0

    assert pd.isna(X.loc[0, 'trestbps'])
    assert X.loc[2, 'trestbps'] == 130.0


def test_split_data():
    X = pd.DataFrame({'feature1': range(100), 'feature2': range(100)})
    y = pd.Series([0] * 50 + [1] * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(y_train) + len(y_val) + len(y_test) == 100
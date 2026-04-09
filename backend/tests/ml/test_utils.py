import pandas as pd
import os
import pytest
from app.ml.utils import save_full_dataset

def test_save_full_dataset(tmp_path):
    train_df = pd.DataFrame({'A': [1, 2], 'target': [0, 1]})
    val_df = pd.DataFrame({'A': [3], 'target': [1]})
    test_df = pd.DataFrame({'A': [4], 'target': [0]})

    out_file = tmp_path / 'full_dataset.csv'

    result_path = save_full_dataset(train_df, val_df, test_df, str(out_file))

    assert os.path.exists(result_path)

    saved_df = pd.read_csv(result_path)

    assert len(saved_df) == 4
    assert list(saved_df.columns) == ['A','target']
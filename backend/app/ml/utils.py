import pandas as pd


def save_full_dataset(train_df, val_df, test_df, out_path):
    train_ready = train_df.copy()
    val_ready = val_df.copy()
    test_ready = test_df.copy()

    train_ready['target'] = train_ready.get('target') if 'target' in train_ready.columns else None
    full_dataset = pd.concat([train_ready, val_ready, test_ready], axis=0)
    full_dataset.to_csv(out_path, index=False)
    return out_path


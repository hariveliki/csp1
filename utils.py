import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler


class F1Score:
    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, y, y_hat):
        y_label = np.unique(y)
        if len(y_label) == 1:
            return 0
        else:
            return f1_score(y, y_hat, average=self.average)


def RandomUnderSampler_(X, y, sampling_strategy='auto', seed=0):
    np.random.seed(seed)
    try:
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
    except Exception as e:
        print(f"RandomUnderSampler failed: {e}\n")
        return None, None
    Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
    np.random.shuffle(Mat)
    Mat = Mat.astype(np.float32)
        
    return Mat[:, :-1], Mat[:, -1]


def load_transpose_CER(file_x, file_case=None, mask='id_pdl', scale_data=False, scaler=StandardScaler()):

    df_data_x = pd.read_csv(file_x, low_memory=False).set_index('index')
    df_data_x.index.name = None
    df_data_x = df_data_x.T
    df_data_x[mask] = df_data_x[mask].astype('int32')
    df_data_x = df_data_x.set_index(mask)
    
    if scale_data:
        df_data_x = pd.DataFrame(scaler.fit_transform(df_data_x.T).T, columns=df_data_x.columns, index=df_data_x.index)
        
    if file_case is not None:
        case = pd.read_csv(file_case).set_index(mask)
        df_data = pd.merge(df_data_x, case, on=mask)
    else:
        df_data = df_data_x.set_index(mask)
    
    return df_data


def create_X_out_of_df(df: pd.DataFrame) -> np.ndarray:
    assert not df.empty, "df is empty"
    assert "Messpunkt_ID" in df.columns, "Messpunkt_ID not in df.columns"
    if "Messpunkt_ID" in df.columns:
        X = df.iloc[:, 1:].values.astype(np.float32)
    return X


def create_y_out_of_df(df: pd.DataFrame) -> np.ndarray:
    assert not df.empty, "df is empty"
    assert "id" in df.columns, "id not in df.columns"
    assert "label" in df.columns, "label not in df.columns"
    y = df.iloc[:, -1:].values.astype(np.float32)
    return y


def create_X_y_out_of_df(df: pd.DataFrame) -> np.ndarray:
    assert not df.empty, "df is empty"
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1:].values.astype(np.float32)
    return X, y


def stack(*args: np.ndarray):
    for arg in args:
        assert isinstance(arg, np.ndarray), f"{arg} is not a np.ndarray"
    shapes = [arg.shape[1:] for arg in args]
    assert len(set(shapes)) == 1, f"shapes are not equal: {shapes}"
    return np.vstack(args)


def check_file_exist(path):
    return os.path.isfile(path)


def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise e
    return path
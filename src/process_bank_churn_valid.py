import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    :param filepath: Path to the CSV file.
    :return: Loaded dataset as a Pandas DataFrame.
    """
    return pd.read_csv(filepath, index_col=0)


def split_data(df: pd.DataFrame, target_col: str = 'Exited', test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Split the dataset into training and validation sets.

    :param df: Raw dataset as a Pandas DataFrame.
    :param target_col: Name of the target column.
    :param test_size: Proportion of the dataset to include in the validation split.
    :param random_state: Random seed for reproducibility.
    :return: Dictionary containing train and validation inputs/targets.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    input_cols = list(train_df.columns)[2:-1]
    train_inputs, train_targets = train_df[input_cols], train_df[target_col]
    val_inputs, val_targets = val_df[input_cols], val_df[target_col]
    
    return {
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'val_inputs': val_inputs,
        'val_targets': val_targets
    }


def preprocess_numeric(data: Dict[str, Any], scale: bool = False) -> None:
    """
    Preprocess numeric columns: apply MinMax scaling if enabled.

    :param data: Dictionary containing train and validation inputs.
    :param scale: Whether to apply MinMax scaling.
    :return: None (modifies the data dictionary in place).
    """
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()[:-1]
    if scale:
        scaler = MinMaxScaler()
        data['train_inputs'][numeric_cols] = scaler.fit_transform(data['train_inputs'][numeric_cols])
        data['val_inputs'][numeric_cols] = scaler.transform(data['val_inputs'][numeric_cols])
    data['numeric_cols'] = numeric_cols


def preprocess_categorical(data: Dict[str, Any]) -> None:
    """
    Apply one-hot encoding to categorical features.

    :param data: Dictionary containing train and validation inputs.
    :return: None (modifies the data dictionary in place).
    """
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    data['train_inputs'][encoded_cols] = encoder.transform(data['train_inputs'][categorical_cols])
    data['val_inputs'][encoded_cols] = encoder.transform(data['val_inputs'][categorical_cols])

    data['categorical_cols'] = categorical_cols
    data['encoded_cols'] = encoded_cols


def process_data(filepath: str, scale: bool = False) -> Dict[str, Any]:
    """
    Main function to load, preprocess, and return train/validation datasets.

    :param filepath: Path to the CSV file.
    :param scale: Whether to apply MinMax scaling to numeric data.
    :return: Dictionary containing processed train and validation datasets.
    """
    raw_df = load_data(filepath)
    data = split_data(raw_df)
    preprocess_numeric(data, scale)
    preprocess_categorical(data)

    X_train = data['train_inputs'][data['numeric_cols'] + data['encoded_cols']]
    X_val = data['val_inputs'][data['numeric_cols'] + data['encoded_cols']]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
    }

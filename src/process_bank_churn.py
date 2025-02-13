import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from typing import Dict, Any, Optional

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    :param filepath: Path to the CSV file.
    :return: Loaded dataset as a Pandas DataFrame.
    """
    return pd.read_csv(filepath, index_col=0)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Split the dataset into training and validation sets.
    :param df: Raw dataset as a Pandas DataFrame.
    :param target_col: Name of the target column.
    :param test_size: Proportion of the dataset to include in the validation split.
    :param random_state: Random seed for reproducibility.
    :return: Dictionary containing train and validation inputs/targets.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    input_cols = list(df.columns)[2:-1]
    return {
        'train_inputs': train_df[input_cols],
        'train_targets': train_df[target_col],
        'val_inputs': val_df[input_cols],
        'val_targets': val_df[target_col]
    }

def preprocess_numeric(data: Dict[str, Any], scale: bool = False, poly_degree: int = 1) -> None:
    """
    Preprocess numeric columns: apply MinMax scaling if enabled and generate polynomial features.
    :param data: Dictionary containing train and validation inputs.
    :param scale: Whether to apply MinMax scaling.
    :param poly_degree: Degree of polynomial features to generate.
    :return: None (modifies the data dictionary in place).
    """
    key = 'train_inputs' if 'train_inputs' in data else 'inputs'
    numeric_cols = data[key].select_dtypes(include=np.number).columns.tolist()

    # Поліноміальні фічі
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    transformed_train = poly.fit_transform(data[key][numeric_cols])

    # Отримуємо нові назви колонок
    new_numeric_cols = poly.get_feature_names_out(numeric_cols).tolist()

    # Присвоюємо нові значення
    data[key] = pd.DataFrame(transformed_train, columns=new_numeric_cols, index=data[key].index)

    if 'val_inputs' in data:
        transformed_val = poly.transform(data['val_inputs'][numeric_cols])
        data['val_inputs'] = pd.DataFrame(transformed_val, columns=new_numeric_cols, index=data['val_inputs'].index)

    # Масштабування, якщо потрібно
    if scale:
        scaler = MinMaxScaler()
        data[key][new_numeric_cols] = scaler.fit_transform(data[key][new_numeric_cols])
        if 'val_inputs' in data:
            data['val_inputs'][new_numeric_cols] = scaler.transform(data['val_inputs'][new_numeric_cols])

    # Оновлюємо список числових колонок
    data['numeric_cols'] = new_numeric_cols

def preprocess_categorical(data: Dict[str, Any]) -> None:
    """
    Apply one-hot encoding to categorical features.
    :param data: Dictionary containing train and validation inputs.
    :return: None (modifies the data dictionary in place).
    """
    key = 'train_inputs' if 'train_inputs' in data else 'inputs'
    categorical_cols = data[key].select_dtypes(include='object').columns.tolist()
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data[key][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    data[key][encoded_cols] = encoder.transform(data[key][categorical_cols])
    if 'val_inputs' in data:
        data['val_inputs'][encoded_cols] = encoder.transform(data['val_inputs'][categorical_cols])
    
    data['categorical_cols'] = categorical_cols
    data['encoded_cols'] = encoded_cols

def process_data(filepath: str, target_col: Optional[str] = None, scale: bool = False, poly_degree: int = 1) -> Dict[str, Any]:
    """
    Main function to load, preprocess, and return train/validation datasets.
    :param filepath: Path to the CSV file.
    :param target_col: Name of the target column (optional, if splitting is needed).
    :param scale: Whether to apply MinMax scaling to numeric data.
    :param poly_degree: Degree of polynomial features to generate.
    :return: Dictionary containing processed train and validation datasets, or just processed dataset.
    """
    raw_df = load_data(filepath)
    
    if target_col:
        data = split_data(raw_df, target_col)
    else:
        data = {'inputs': raw_df[list(raw_df.columns)[2:]]
}
    
    preprocess_numeric(data, scale, poly_degree)
    preprocess_categorical(data)
    
    key = 'train_inputs' if 'train_inputs' in data else 'inputs'
    X = data[key][data['numeric_cols'] + data['encoded_cols']]
    
    if 'train_inputs' in data:
        X_val = data['val_inputs'][data['numeric_cols'] + data['encoded_cols']]
        return {
            'train_X': X,
            'train_y': data['train_targets'],
            'val_X': X_val,
            'val_y': data['val_targets'],
        }
    return {'X': X}

import pandas as pd
import os

def load_and_preprocess(file_path):
    """
    Load and preprocess a CSV file.
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    data = pd.read_csv(file_path)

    # Preprocessing: Fill missing values and parse timestamps
    if 'TimeStamp' in data.columns:
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
    if 'OrderPrice' in data.columns:
        data['OrderPrice'] = data['OrderPrice'].fillna(method='ffill')

    return data

def merge_files_in_training_data(directory):
    """
    Merge all CSV files in the TrainingData directory into a single DataFrame.
    Args:
        directory (str): Path to the TrainingData directory.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    dataframes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                dataframes.append(load_and_preprocess(file_path))
    return pd.concat(dataframes, ignore_index=True)

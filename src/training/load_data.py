import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path: str, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads the processed flight data and splits it into train and test sets.

    Args:
        data_path (str): Path to the processed CSV dataset.
        random_state (int): Random seed for reproducible splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test datasets.

    Dependencies:
        - os.path.exists
        - pandas.read_csv
        - sklearn.model_selection.train_test_split
    """
    print(f"Loading processed data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Define features and target
    features = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 'DEPARTURE_DELAY']
    target = 'delayed'
    
    X = df[features]
    y = df[target]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

import pandas as pd
import pytest
from src.preprocessing.reduce_data import clean_dataframe

def test_clean_dataframe_removes_cancelled_and_diverted():
    # Setup mock data with cancelled, diverted and valid flights
    data = {
        'MONTH': [1, 2, 3, 4],
        'DAY_OF_WEEK': [1, 2, 3, 4],
        'AIRLINE': ['AA', 'DL', 'UA', 'WN'],
        'DISTANCE': [500, 600, 700, 800],
        'ARRIVAL_DELAY': [10.0, 20.0, 5.0, 12.0],
        'DEPARTURE_DELAY': [5.0, 15.0, 0.0, 8.0],
        'CANCELLED': [0, 1, 0, 0], # Flight 2 is cancelled
        'DIVERTED': [0, 0, 1, 0]   # Flight 3 is diverted
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_dataframe(df)
    
    # Check that only flights 1 and 4 remain (cancelled and diverted dropped)
    assert len(cleaned_df) == 2
    assert cleaned_df.iloc[0]['AIRLINE'] == 'AA'
    assert cleaned_df.iloc[1]['AIRLINE'] == 'WN'

def test_clean_dataframe_creates_correct_target():
    # Setup mock data with delays > 15 and <= 15
    data = {
        'MONTH': [1, 2, 3],
        'DAY_OF_WEEK': [1, 2, 3],
        'AIRLINE': ['AA', 'DL', 'UA'],
        'DISTANCE': [500, 600, 700],
        'ARRIVAL_DELAY': [15.0, 16.0, -5.0], # 15 should be delayed=0, 16 delayed=1, -5 delayed=0
        'DEPARTURE_DELAY': [10.0, 12.0, -2.0],
        'CANCELLED': [0, 0, 0],
        'DIVERTED': [0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_dataframe(df)
    
    assert len(cleaned_df) == 3
    # delayed: ARRIVAL_DELAY > 15 -> 0, 1, 0
    assert cleaned_df.iloc[0]['delayed'] == 0
    assert cleaned_df.iloc[1]['delayed'] == 1
    assert cleaned_df.iloc[2]['delayed'] == 0

def test_clean_dataframe_drops_missing_values():
    # Setup mock data with missing values in ARRIVAL_DELAY or other features
    data = {
        'MONTH': [1, 2, None], # Month missing on row 3
        'DAY_OF_WEEK': [1, 2, 3],
        'AIRLINE': ['AA', 'DL', 'UA'],
        'DISTANCE': [500, 600, 700],
        'ARRIVAL_DELAY': [10.0, None, 5.0], # ARRIVAL_DELAY missing on row 2
        'DEPARTURE_DELAY': [5.0, 15.0, 0.0],
        'CANCELLED': [0, 0, 0],
        'DIVERTED': [0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_dataframe(df)
    
    # Only row 1 is fully complete and valid
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]['AIRLINE'] == 'AA'

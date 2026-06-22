import argparse
import os
import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filters cancelled/diverted flights, creates target binary column, and cleans missing values."""
    # Filter out cancelled and diverted flights
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    # Drop rows where target variable is missing
    df = df.dropna(subset=['ARRIVAL_DELAY'])

    # Construct the binary target: delayed = 1 if ARRIVAL_DELAY > 15, else 0
    df['delayed'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

    # Drop unnecessary columns that were used for filtering or constructing target
    df = df.drop(columns=['CANCELLED', 'DIVERTED', 'ARRIVAL_DELAY'])

    # Drop any remaining missing values in features
    df = df.dropna()
    return df

def preprocess_data(input_path: str, output_path: str, sample_size: int):
    print(f"Reading raw data from {input_path}...")
    
    # Load only necessary columns to optimize memory usage
    cols_to_use = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 
        'ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'CANCELLED', 'DIVERTED'
    ]
    
    # Check if file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data file not found at: {input_path}")
        
    df = pd.read_csv(input_path, usecols=cols_to_use)
    print(f"Loaded {len(df)} rows.")

    print("Cleaning dataframe...")
    df = clean_dataframe(df)
    print(f"Remaining flights after cleaning: {len(df)}")

    # Handle sampling to avoid memory issues during training
    if sample_size > 0 and sample_size < len(df):
        print(f"Sampling {sample_size} rows...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled dataset count: {len(df)}")

    # Display class balance
    class_balance = df['delayed'].value_counts(normalize=True) * 100
    print(f"Class balance (delayed):")
    print(f"  0 (No delay / delay <= 15 min): {class_balance.get(0, 0.0):.2f}%")
    print(f"  1 (Delay > 15 min): {class_balance.get(1, 0.0):.2f}%")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess flight delay raw data.")
    parser.add_argument('--input-path', type=str, default='data/raw/flights.csv',
                        help='Path to the raw flights.csv file')
    parser.add_argument('--output-path', type=str, default='data/processed/processed_flights.csv',
                        help='Path where the processed csv will be saved')
    parser.add_argument('--sample-size', type=str, default='100000',
                        help='Number of rows to sample (default 100000; set to "all" to disable sampling)')
    
    args = parser.parse_args()
    
    # Parse sample-size parameter
    if args.sample_size.lower() == 'all':
        sample_size = -1
    else:
        try:
            sample_size = int(args.sample_size)
        except ValueError:
            print("Invalid sample-size value. Using default of 100000.")
            sample_size = 100000
            
    preprocess_data(args.input_path, args.output_path, sample_size)

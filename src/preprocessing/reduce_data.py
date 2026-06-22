import pandas as pd

df_sample = pd.read_csv("data/raw/flights.csv", nrows=200000)

df_sample.to_csv("data/raw/sample_200k.csv", index=False)
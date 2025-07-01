import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/ENB2012_nozero.csv')

print("Dataset shape:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)

print("\nMin values for each column:")
print(df.min())

print("\nMax values for each column:")
print(df.max())

print("\nChecking for problematic values:")
print("Any negative values:", (df < 0).any().any())
print("Any zero values:", (df == 0).any().any())
print("Any NaN values:", df.isna().any().any())
print("Any infinite values:", np.isinf(df.select_dtypes(include=[np.number])).any().any())

print("\nSample of first few rows:")
print(df.head())

print("\nSample of last few rows:")
print(df.tail()) 
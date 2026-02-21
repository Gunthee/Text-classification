import pandas as pd

df = pd.read_csv('training_dataset_cleaned_balanced.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.shape)

df = df.dropna()
print('After dropping null values:')
print(df.isnull().sum())
print(df.shape)
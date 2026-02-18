import pandas as pd

df = pd.read_csv('thaimooc_courses.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.shape)

df = df.dropna()
print('After dropping null values:')
print(df.isnull().sum())
print(df.shape)
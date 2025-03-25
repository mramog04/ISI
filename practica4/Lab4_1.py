import pandas as pd
import numpy as np
import matplotlib as plt

info = pd.read_csv('data.csv')
print(info.head())
""" print(info.head(5))"""

print(info.dtypes)
print(info.isnull())
""" print(info.isnull().sum()) """

print("Media:")
print(info.iloc[:,2:].mean())
print("Mediana:")
print(info.iloc[:,2:].median())

df = info[info['Age'] > 20]
print(df)

edad = info.groupby('Age')
print(edad.mean())

df.to_csv('filtrado_por_edad.csv', index=False)
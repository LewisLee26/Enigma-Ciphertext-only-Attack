import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


parquet_file = 'dataset/germanEncryptionRaw.parquet'
df = pd.read_parquet(parquet_file, engine='auto')

count = np.ulonglong(0)
df['Length'] = 0

index = 0
for i in df['Text']:
    df.loc[index, 'Length'] = len(i)
    count += len(i)
    index += 1

df = df.sort_values(by='Length', ascending=True)
df = df.reset_index()
df = df.drop('index', axis=1)

print(df.head())

print("size")
print(df.size)


print(len(df[(df['Text']==0)]))

df = df.drop(df[df.Length < 100].index)
df = df.reset_index()

df = df[df.Encryption == False]
df = df.reset_index()

print(df.head())

print("size")
print(df.size)

print(df.iloc[-1].Text)

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

print(len(df[(df['Text']==0)]))

# plt.plot(df.index,df.loc[:,'Length'])
# plt.ylabel('Length')
# plt.xlabel('Index')
# plt.show()

## remove texts of length 0
## split texts of length 2500 or more into multiple texts

df = df.drop(df[df.Length < 5].index)
df = df.reset_index()
# df = df.drop(df[df.Length > 1000].index)


def splitText(dataframe):
    max_length = 1000
    new_dataframe = dataframe.copy()

    for index, row in dataframe.iterrows():
        if row['Length'] > max_length:
            temp_text = row['Text'][max_length:]
            new_dataframe.at[index, 'Text'] = row['Text'][:max_length]
            new_dataframe.at[index, 'Length'] = max_length
            new_dataframe = pd.concat([new_dataframe, pd.DataFrame([{'Text':temp_text, 'Encryption':row['Encryption'], 'Length':len(temp_text)}])], ignore_index=True)

    return new_dataframe


while True:
    newdf = splitText(df)
    if newdf.shape[0] == df.shape[0]:
        break
    else:
        df = newdf

df = df.sort_values(by='Length', ascending=True)
df = df.reset_index()
df = df.drop('index', axis=1)

plt.plot(df.index,df.loc[:,'Length'])
plt.ylabel('Length')
plt.xlabel('Index')
plt.show()

table = pa.Table.from_pandas(df)
pq.write_table(table, 'dataset/germanEncryption.parquet')


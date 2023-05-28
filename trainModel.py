import torch
import numpy as np
import wandb
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="enigmaCOA",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "Transformer",
#     "dataset": "enigmaData",
#     "epochs": 10,
#     }
# )

# hyper Parameters

classes = ('decrypted', 'encrypted')


class CustomTextDataset(torch.utils.data.Dataset):
  def __init__(self, dataframe, transform=None):
    self.df = dataframe
    self.transform = transform
    self.x = self.df.Text
    self.y = self.df.Encryption
    self.n_samples = len(self.df)

  def __len__(self):
    return self.n_samples

  def __getitem__(self, index):
    return self.x[index], self.y[index]


  def returnData(self):
    return self.x, self.y

  # change it so that the size of the tensor fits the size of the text
  # maybe have it as a 1d tensor
  def char_to_num(self):
    # max_length = self.x.str.len().max()
    max_length = 32768 # 2^15
    length_x = len(self.x)
    # result = np.zeros((length_x, max_length.astype(int)), dtype=int)
    result = np.zeros((length_x, max_length), dtype=int)
    for i, string in enumerate(self.x):
      string = str(string)
      for j, char in enumerate(string):
        result[i, j] = ord(char) - ord('a')

    result_tensor = torch.tensor(result)
    self.x = result_tensor

  def binaryLabels(self):
    for i in range(len(self.y)):
      if self.y.iloc[i] == True:
        self.y.loc[self.y.index[i]] = 1
      else:
        self.y.loc[self.y.index[i]] = 0

    self.y = torch.tensor(self.y)



parquet_file = 'germanEncryption.parquet'
df = pd.read_parquet(parquet_file, engine='auto')
dataset = CustomTextDataset(df)

dataset.char_to_num()
dataset.binaryLabels()

x_train, x_test, y_train, y_test = train_test_split(dataset.returnData()[0], dataset.returnData()[1], test_size=0.2, random_state=1234)



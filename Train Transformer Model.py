import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper Parameters
architecture = "transformer"
dataset = "germanEncryption"
group = "transformer"
num_epochs = 100
batch_size = 32
sequence_length = 1000

# maybe change classes to ['encrypted', 'decrypted']
classes = ['Encrypted']
num_classes = len(classes)

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="enigmaCOA",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": learning_rate,
#     "architecture": architecture,
#     "dataset": dataset,
#     "group": group,
#     "epochs": num_epochs,
#     "learning_rate": learning,
#     "num_layers": num_layers,
#     "batch_size": batch_size,
#     "hidden_size": hidden_size,
#     "input_size": input_size,
#     "sequence_length": sequence_length
#     }
# )

model_name = 'bert-base-uncased'

class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        self.x = self.df.Text.copy()
        self.y = self.df.Encryption.copy()
        self.n_samples = len(self.df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def returnData(self):
        return self.x, self.y

    def binaryLabels(self):
        self.y.loc[(self.y == True)] = 1
        self.y.loc[(self.y == False)] = 0
        self.y = torch.tensor(self.y)
    
    # maybe change to number instead of tensor
    def char_to_tensor(self):
        max_length = sequence_length
        length_x = len(self.x)

        result = np.zeros((length_x, max_length, 1, 26), dtype=int)
        for i, string in enumerate(self.x):
            string = str(string)
            for j, char in enumerate(string):
                result[i, j, 0, ord(char) - ord('a')] = 1
    
        result_tensor = torch.tensor(result)
        self.x = result_tensor


parquet_file = 'dataset/germanEncryption.parquet'
df = pd.read_parquet(parquet_file, engine='auto')
dataset = CustomTextDataset(df)

dataset.binaryLabels()
dataset.char_to_tensor()

x_train, x_test, y_train, y_test = train_test_split(dataset.returnData()[0], dataset.returnData()[1], test_size=0.2, random_state=1234)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


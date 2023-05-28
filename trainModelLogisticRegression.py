import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb
import pandas as pd

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

num_epochs = 200,
learning_rate = 0.01,
architecture = "LogisticRegression",
dataset = "germanEncryption"

wandb.init(
    # set the wandb project where this run will be logged
    project="enigmaCOA",
    
    # track hyperparameters and run metadata
    config={
        num_epochs,
        learning_rate,
        architecture,
        dataset  
    }
)

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

    def char_to_num(self):
        max_length = 1000 
        length_x = len(self.x)
        result = np.zeros((length_x, max_length), dtype=int)
        for i, string in enumerate(self.x):
            string = str(string)
        for j, char in enumerate(string):
            result[i, j] = ord(char) - ord('a')

        result_tensor = torch.tensor(result)
        self.x = result_tensor

    def binaryLabels(self):
        self.y.loc[(self.y == True)] = 1
        self.y.loc[(self.y == False)] = 0

        self.y = torch.tensor(self.y)

parquet_file = 'dataset/germanEncryption.parquet'
df = pd.read_parquet(parquet_file, engine='auto')   
dataset = CustomTextDataset(df)

dataset.char_to_num()
dataset.binaryLabels()

x_train, x_test, y_train, y_test = train_test_split(dataset.returnData()[0], dataset.returnData()[1], test_size=0.2, random_state=1234)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicited = torch.sigmoid(self.linear(x))
        return y_predicited
  
n_features = 1000
model = LogisticRegression(n_features)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

example_ct = 0
for epoch in range(num_epochs):
    example_ct+=1
    y_predicted = model(x_train)
    loss = criterion(y_predicted, y_train.float())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # wandb.log({'epoch': epoch+1, 'loss': loss.item()}, step=example_ct)
    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')

torch.save(model.state_dict(), 'models/model.pt')
wandb.save('model.pt')

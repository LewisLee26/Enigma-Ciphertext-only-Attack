import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper Parameters
architecture = "RNN"
dataset = "germanEncryption"
group = "RNN"
num_epochs = 5
learning_rate = 0.001
num_layers = 2
batch_size = 1
hidden_size = 256
input_size = 26
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
#     }
# )


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
  
    def char_to_tensor(self):
        max_length = 1000
        length_x = len(self.x)

        result = np.zeros((length_x, max_length, 1, 26), dtype=int)
        for i, string in enumerate(self.x):
            string = str(string)
            for j, char in enumerate(string):
                result[i, j, 0, ord(char) - ord('a')] = 1
    
        result_tensor = torch.tensor(result)
        self.x = result_tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = combined.to(device)
        hidden = self.i2h(combined).clone()
        output = self.i2o(combined)
        # output = self.softmax(output)
        output = torch.sigmoid(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).clone()


parquet_file = 'dataset/germanEncryption.parquet'
df = pd.read_parquet(parquet_file, engine='auto')
dataset = CustomTextDataset(df)

dataset.binaryLabels()
dataset.char_to_tensor()

x_train, x_test, y_train, y_test = train_test_split(dataset.returnData()[0], dataset.returnData()[1], test_size=0.2, random_state=1234)
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

rnn = RNN(input_size, hidden_size, num_classes).to(device)

def class_from_output(output):
    prediction = output.round()
    return prediction

criterion =  nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

start_time = time.time()
for epoch in range(num_epochs):
    predictions_total = 0
    loss_total = 0
    for i in range(round(x_train.size()[0])):
        hidden = rnn.init_hidden().to(device)
        # for j in range(x_train[i].size()[0]):
        # trying to stop vanishing gradient
        for j in range(100):
            if torch.count_nonzero(x_train[i][j]) == 0:
                break
            y_prediction, hidden = rnn(x_train[i][j], hidden)

        loss = criterion(y_prediction[0][0], y_train[i].float())
        loss_total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            end_time = time.time()
            print(f'\rEpoch [{epoch+1}/{num_epochs}] - [{i+1}/{x_train.size()[0]}] - loss: [{round((loss_total)/(i+1),4)}] - Time [{(end_time-start_time):.4f}s]',end='')
    scheduler.step()
    print()

    # wandb.log({'loss': (loss_total/split)/(i+1)}, step=epoch+1)

with torch.no_grad():
    correct = 0
    for i in range(round(x_test.size()[0]/10)):
        hidden = rnn.init_hidden().to(device)
        for j in range(x_test[i].size()[0]):
            if torch.count_nonzero(x_test[i][j]) == 0:
                break
            y_prediction, hidden = rnn(x_test[i][j], hidden)
        correct += int(torch.round(torch.sigmoid(y_prediction[0][0])) == y_test[i].float())
        print(f'\r{i+1}/{round(x_test.size()[0]/10)} - acc: {round((correct)/(i+1),4)}',end='')
    print(correct)
    print()

torch.save(rnn.state_dict(), 'models/RNNModel.pt')

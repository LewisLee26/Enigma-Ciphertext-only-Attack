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
group = "RNN(batch)"
num_epochs = 100
learning_rate = 0.001
num_layers = 2
batch_size = 8
hidden_size = 128
input_size = 26
sequence_length = 1000
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
        max_length = sequence_length
        length_x = len(self.x)

        result = np.zeros((length_x, max_length, 1, 26), dtype=int)
        for i, string in enumerate(self.x):
            string = str(string)
            for j, char in enumerate(string):
                result[i, j, 0, ord(char) - ord('a')] = 1
    
        result_tensor = torch.tensor(result)
        self.x = result_tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        input = input.to(torch.float32).to(device)
        out, _ = self.rnn(input, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


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

rnn = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# examples = iter(train_loader)
# samples, labels = next(examples)
# print(samples.shape, labels.shape)

def class_from_output(output):
    prediction = output.round()
    return prediction

criterion =  nn.BCEWithLogitsLoss()
# criterion =  nn.BCELoss()
# criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, weight_decay=5e-5)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=5e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

total_steps = len(train_loader)
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    total_loss = 0
    start_time = time.time()
    for i, (x, y) in enumerate(train_loader):
        x = x.reshape(-1, sequence_length, input_size).to(device)
        y = y.to(device)
        y_prediction = rnn(x)

        loss = criterion(y_prediction.reshape(-1).float(), y.reshape(-1).float())
        total_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        time.sleep(1)
        optimizer.step()

        if (i+1) % 10 == 0:
            end_time = time.time()
            print (f'\rEpoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Time [{(end_time-start_time):.4f}s], Time/Step [{((end_time-start_time)/total_steps):.4f}s], Loss: {(total_loss/(i+1)):.4f}',end='')

    # scheduler.step()
    print()
    
    # wandb.log({'loss': (loss_total/split)/(i+1)}, step=epoch+1)

with torch.no_grad():
    num_correct = 0
    num_samples = 0
    predictions_total = 0
    loss_total = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.reshape(-1, sequence_length, input_size).to(device)
        y = y.to(device)
        y_prediction = rnn(x)
        
        num_samples += y.size(0)
        num_correct += (torch.round(y_prediction.reshape(-1).float()) == y.reshape(-1).float()).sum().item()

    for p in rnn.parameters():
         print(p.grad)
    
    acc = 100.0 * num_correct / num_samples
    print(f'Accuracy: {acc} %')

torch.save(rnn.state_dict(), 'models/RNNModel.pt')

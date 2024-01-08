import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import torch.onnx
import wandb
import os
import json

# Define the size of the character set
CHARSET_SIZE = 27 # A-Z + empty token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=20)


# Model for classifiying text
class TransformerModel(nn.Module):

    def __init__(self, nhead, adim, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.adim = adim
        # encoder_layers = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout, batch_first = True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout, batch_first = True)

        self.classifier = nn.Linear(nhid*adim, 1)

    def forward(self, src):
        src = F.one_hot(src.to(torch.int64), CHARSET_SIZE).float()
        src = self.resize_embedding(src)
        src = self.transformer_encoder(src)
        src = torch.flatten(src, -2)
        
        # don't use a sigmoid function on the output inside the mode
        src = self.classifier(src)
        return src

    def resize_embedding(self, input_tensor):
        input_size = list(input_tensor.shape)
        input_size[-1] = self.adim - CHARSET_SIZE
        zeros_tensor = torch.zeros(input_size, dtype=input_tensor.dtype, device=input_tensor.device)
        output_tensor = torch.cat([input_tensor, zeros_tensor], dim=-1)

        return output_tensor


def train(model, criterion, optimizer, epochs, dataloader):
    pbar = tqdm(total=epochs*len(dataloader), desc="Traning - Epoch: 1 - Loss: None")
    checkpoint = 1
    for epoch in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(dataloader):
            model.train()
            outputs = model(inputs.to(device))
            # the sigmoid is used outisde the model during training and testing
            # this is so it can be used as a regression model in deployment
            outputs = F.sigmoid(outputs.to(dtype=torch.float64)).squeeze(-1)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss, epoch)

            running_loss += loss.item()
            pbar.set_description(f"Training - Epoch: {epoch+1} - Loss: {running_loss/(i+1):.5f}")
            pbar.update(1)

            wandb.log({"training_loss": loss.item(), "training_mean_running_loss": running_loss/(i+1)})

            if (i % (len(dataloader) // checkpoints_p_epoch) == 0 and i != 0) or i == 0 :
                
                average_loss, accuracy = test(model, test_dataloader_trimmed, criterion)
                wandb.log({"test_mean_running_loss": average_loss, "test_accuracy": accuracy})
                
                if i % (len(dataloader) // checkpoints_p_epoch) == 0 and i != 0:
                    torch.save(model.state_dict(), f"{folder_path}/checkpoint_{checkpoint}.pt")
                    checkpoint += 1

        average_loss, accuracy = test(model, test_dataloader, criterion)
        wandb.log({"test_mean_running_loss": average_loss, "test_accuracy": accuracy})
        print({"test_mean_running_loss": average_loss, "test_accuracy": accuracy})

        torch.save(model.state_dict(), f"{folder_path}/checkpoint_{checkpoint}.pt")
        checkpoint += 1

def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            outputs = F.sigmoid(outputs).squeeze(-1)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            # Assuming binary classification
            predicted_labels = torch.sigmoid(outputs).round()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)


    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        label = self.labels[idx]

        return token, label


# loading the dataframe
df = pd.read_parquet(r"dataset\enigma_binary_classification_wiki_en_0-13_plugs.parquet")

# Tokenizing the dataframe
text_list = []
input_size = 512
for text in tqdm(df["text"], desc="Loading dataframe"):
    tokens = torch.tensor([ord(char) - ord('A') + 1 for char in (list(text))], dtype=torch.int)
    if 512-tokens.size(0) > 0:
        zeros = torch.zeros(512-tokens.size(0), dtype=torch.int)
        tokens = torch.concatenate((tokens, zeros))
    text_list.append(tokens)
df['text'] = text_list

# Converting the dataframe into a dataset
test_split = 0.2
test_split_index = round((1-test_split)*len(df['text']))
print("Converting dataframe to dataset")
train_dataset = CustomDataset(df['text'][:test_split_index].to_list(), df['label'][:test_split_index].to_list())
test_dataset = CustomDataset(df['text'][test_split_index:].to_list(), df["label"][test_split_index:].to_list())
# A smaller dataset for testing mid-epoch
test_dataset_trimmed = CustomDataset(df['text'][test_split_index:test_split_index+500].to_list(), df["label"][test_split_index:test_split_index+500].to_list())
print("Converting dataframe to dataset complete")


# Create dataloaders
batch_size = 10
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_dataloader_trimmed = DataLoader(
    test_dataset_trimmed,
    batch_size=batch_size,
    shuffle=False
)


# Instantiate the model with some sample parameters
nhead = 3
nhid = 512
nlayers = 1
dropout = 0.1
lr = 5e-4
epochs = 15
adim = ((CHARSET_SIZE // nhead) + 1) * nhead

# Instantiate the model
model = TransformerModel(nhead, adim, nhid, nlayers, dropout).to(device)

criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust step_size and gamma as needed
# scheduler = ReduceLROnPlateau(optimizer)

# Create a folder for the model
checkpoints_p_epoch = 5
folder_limit = 1000
folder_created = False
for i in range(folder_limit):
    folder_path = f"model/model_{i+1}"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        folder_created = True
        print(f"Folder '{folder_path}' created successfully.")
        break

# Run the training if a model folder is created 
if folder_created == True:
    # Specify the path for the JSON file within the folder
    json_file_path = os.path.join(folder_path, 'config.json')

    # Model config from parameters
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "num_heads": nhead,
        "hidden_dimensions": nhid,
        "num_layers": nlayers,
        "dropout": dropout,
    }

    # Write the JSON data to the file
    with open(json_file_path, 'w') as json_file:
        json.dump(config, json_file, indent=2)

    print(f"JSON file '{json_file_path}' created successfully.")


    # Initialise wandb run
    wandb.init(
        project="Enigma Ciphertext-only Attack",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "num_heads": nhead,
            "hidden_dimensions": nhid,
            "num_layers": nlayers,
            "dropout": dropout,
        }
    )


    train(model, criterion, optimizer, epochs=epochs, dataloader=train_dataloader)

    test(model, test_dataloader, criterion)


    # torch.save(model.state_dict(), r"model/model_7.pt")


    # dummy_input = torch.rand(512).to(torch.int)

    # torch.onnx.export(model,               # model being run
    #                 dummy_input,                         # model input (or a tuple for multiple inputs)
    #                 r"model/model_7.onnx",   # where to save the model (can be a file or file-like object)
    #                 verbose=True
    #                 )

else:
    print(f"Folder limit ({folder_limit}) readed ")
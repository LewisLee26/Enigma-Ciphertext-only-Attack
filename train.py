import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from tqdm import tqdm
import torch.onnx
import wandb

# Define the size of the character set
# 28 is a multiple of 4
CHARSET_SIZE = 26  # A-Z (2  empty embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=20)


# Model for classifiying text
class TransformerModel(nn.Module):

    def __init__(self, nhead, adim, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.adim = adim
        encoder_layers = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

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
    for epoch in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(dataloader):
            model.train()
            outputs = model(inputs)
            # the sigmoid is used outisde the model during training and testing
            # this is so it can be used as a regression model in deployment
            outputs = F.sigmoid(outputs.to(dtype=torch.float64)).squeeze(-1)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(f"Training - Epoch: {epoch+1} - Loss: {running_loss/(i+1):.5f}")
            pbar.update(1)

            wandb.log({"training_loss": loss.item(), "training_mean_running_loss": running_loss/(i+1)})

            if i % (len(dataloader) // 5) == 0:
                average_loss, accuracy = test(model, test_dataloader_trimmed, criterion)
                wandb.log({"test_mean_running_loss": average_loss, "test_accuracy": accuracy})


        average_loss, accuracy = test(model, test_dataloader, criterion)
        wandb.log({"test_mean_running_loss": average_loss, "test_accuracy": accuracy})
        print({"test_mean_running_loss": average_loss, "test_accuracy": accuracy})


def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
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


test_dataset = load_from_disk(r"dataset\enigma_binary_classification_en_0-13_plugs\test")
test_tokens = [torch.tensor([ord(char) - 65 for char in (list(text))]) for text in tqdm(test_dataset['text'], desc="Loading Testing Dataset")]   
test_labels = torch.tensor(test_dataset['label'])
test_dataset = CustomDataset(test_tokens, test_labels)
test_dataset_trimmed = CustomDataset(test_tokens[:500], test_labels[:500])


train_dataset = load_from_disk(r"dataset\enigma_binary_classification_en_0-13_plugs\train")
train_tokens = [torch.tensor([ord(char) - 65 for char in (list(text))]) for text in tqdm(train_dataset['text'], desc="Loading Training Dataset")]   
train_labels = torch.tensor(train_dataset['label'])
train_dataset = CustomDataset(train_tokens, train_labels)

batch_size = 15

# Example DataLoader for training
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Example DataLoader for validation
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
nhead = 1
nhid = 512
nlayers = 1
dropout = 0.1
lr = 0.0005
epochs = 2
adim = CHARSET_SIZE + CHARSET_SIZE % nhead

# Instantiate the model
model = TransformerModel(nhead, adim, nhid, nlayers, dropout)
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr)




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


torch.save(model.state_dict(), r"model/model_4.pt")


dummy_input = torch.rand(512).to(torch.int64)

torch.onnx.export(model,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  r"model/model_4.onnx",   # where to save the model (can be a file or file-like object)
                  verbose=True
                  )
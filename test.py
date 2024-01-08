import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

CHARSET_SIZE = 27

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

def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
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


# loading the dataframe
df = pd.read_parquet(r"dataset\enigma_binary_classification_wiki_en_4_plugs.parquet")

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
test_dataset = CustomDataset(df['text'][test_split_index:].to_list(), df["label"][test_split_index:].to_list())
print("Converting dataframe to dataset complete")

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# Instantiate the model with some sample parameters
nhead = 3
nhid = 512
nlayers = 1
dropout = 0.1
adim = ((CHARSET_SIZE // nhead) + 1) * nhead

# Instantiate the model
model = TransformerModel(nhead, adim, nhid, nlayers, dropout)
criterion = BCEWithLogitsLoss()

# model.load_state_dict(torch.load(r"model/model_7.pt"))
model.load_state_dict(torch.load(r"model\model_1\checkpoint_12.pt"))
model.eval()

print(test(model, test_dataloader, criterion))

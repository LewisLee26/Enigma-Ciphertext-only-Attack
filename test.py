import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm

CHARSET_SIZE = 26

class TransformerModel(nn.Module):

    def __init__(self, nhead, adim, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.adim = adim
        encoder_layers = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.classifier = nn.Linear(nhid*adim, 1)

    def forward(self, src):
        src = F.one_hot(src, CHARSET_SIZE).float()
        src = self.resize_embedding(src)
        src = self.transformer_encoder(src)
        src = torch.flatten(src, -2)
        src = self.classifier(src)
        return F.sigmoid(src).squeeze(-1)

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
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            # Assuming binary classification
            predicted_labels = torch.sigmoid(outputs).round()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)


    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

test_dataset = load_from_disk(r"dataset\enigma_binary_classification_en_0-13_plugs\test")
test_tokens = [torch.tensor([ord(char) - 65 for char in (list(text))]).to(torch.int64) for text in tqdm(test_dataset['text'], desc="Loading Testing Dataset")]   
test_labels = torch.tensor(test_dataset['label'])
test_dataset = CustomDataset(test_tokens, test_labels)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

nhead = 1
nhid = 512
nlayers = 1
dropout = 0.1
adim = CHARSET_SIZE + CHARSET_SIZE % nhead

# Instantiate the model
model = TransformerModel(nhead, adim, nhid, nlayers, dropout)
criterion = BCEWithLogitsLoss()

model.load_state_dict(torch.load(r"model/model_0.pt"))
model.eval()

print(test(model, test_dataloader, criterion))

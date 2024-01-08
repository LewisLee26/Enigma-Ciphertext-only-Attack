
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from model import VerboseTransformerModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from captum.attr import visualization as viz
# from captum.attr import LayerConductance, LayerIntegratedGradients

import torchlens as tl

CHARSET_SIZE = 27

class TransformerModel(nn.Module):

    def __init__(self, nhead, adim, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.adim = adim
        # encoder_layers = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout)

        self.classifier = nn.Linear(nhid*adim, 1)

    def forward(self, src):
        src = F.one_hot(src, CHARSET_SIZE).float()
        src = self.resize_embedding(src)
        src = self.transformer_encoder(src)
        src = torch.flatten(src, -2)
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

nhead = 3
nhid = 512
nlayers = 1
dropout = 0.1
adim = CHARSET_SIZE + CHARSET_SIZE % nhead

model = VerboseTransformerModel(nhead, adim, nhid, nlayers, dropout, True)

path = r"model/model_1_12.onnx"
model.eval()

test_input = test_dataset.__getitem__(101)[0]

# print(test_input)
# model_history = tl.log_forward_pass(model, test_input, layers_to_save='all', vis_opt='none')
# print(model_history)
# print(model.transformer_encoder.self_attn.out_proj.weight)

trim_size = 100
test_input = test_input[:trim_size]
zeros = torch.zeros(512-trim_size, dtype=torch.int)
test_input = torch.concatenate((test_input, zeros), dim=0)

all_tokens = [chr(i+65) for i in test_input.tolist()]

output, attention_weight = model(test_input)

def visualize_token2head_scores(index, x, y):
    fig = plt.figure(figsize=(30, 50))

    for idx in range(x * y):
        test_input = test_dataset.__getitem__(index+idx)[0]

        trim_size = 100
        test_input = test_input[:trim_size]
        zeros = torch.zeros(512-trim_size, dtype=torch.int)
        test_input = torch.concatenate((test_input, zeros), dim=0)


        all_tokens = [chr(i+65) for i in test_input.tolist()]
        _, attention_weight = model(test_input)

        scores_np = attention_weight.detach().numpy()
        scores_mat = scores_np.tolist()

        ax = fig.add_subplot(x, y, idx+1)

        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(scores_mat)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores_mat[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx+1))

    fig.colorbar(im, fraction=0.046, pad=0.04)
    # plt.tight_layout()
    plt.show()

visualize_token2head_scores(6, 2, 2)
    
import onnx
import onnxruntime
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

path = r"model/model_1_12.onnx"


onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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
    # model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader):
        ort_inputs = {model.get_inputs()[0].name: to_numpy(inputs[0])}
        outputs = model.run(None, ort_inputs)
        # print(torch.tensor(outputs[0][0]).float(), labels[0].float())
        loss = criterion(torch.tensor(outputs[0][0]).float(), labels[0].float())
        total_loss += loss.item()

        # Assuming binary classification
        predicted_labels = torch.sigmoid(torch.tensor(outputs[0])).round()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)


    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

# loading the dataframe
df = pd.read_parquet(r"dataset\enigma_binary_classification_wiki_en_12_plugs.parquet")

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

criterion = BCEWithLogitsLoss()

print(test(ort_session, test_dataloader, criterion))
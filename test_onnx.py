import onnx
import onnxruntime
import torch
from tqdm import tqdm
import numpy as np

path = r"model/model_0.onnx"

onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.rand(512).to(torch.int64)

# compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
# print(ort_outs)
# input()

# for i in tqdm(range(100000)):
#     ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk



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
        loss = criterion(torch.tensor(outputs[0]).float(), labels[0].float())
        total_loss += loss.item()

        # Assuming binary classification
        predicted_labels = torch.sigmoid(torch.tensor(outputs[0])).round()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)


    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

test_dataset = load_from_disk(r"dataset\enigma_binary_classification_en_0_plugs\test")
test_tokens = [torch.tensor([ord(char) - 65 for char in (list(text))]).to(torch.int64) for text in tqdm(test_dataset['text'], desc="Loading Testing Dataset")]   
test_labels = torch.tensor(test_dataset['label'])
test_dataset = CustomDataset(test_tokens, test_labels)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


criterion = BCEWithLogitsLoss()


print(test(ort_session, test_dataloader, criterion))
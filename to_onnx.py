import torch 
import torch.nn as nn
import torch.nn.functional as F

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

# Instantiate the model with some sample parameters
nhead = 3
nhid = 512
nlayers = 2
dropout = 0.1
lr = 5e-4
epochs = 1
adim = ((CHARSET_SIZE // nhead) + 1) * nhead

# Instantiate the model
model = TransformerModel(nhead, adim, nhid, nlayers, dropout)

# model.load_state_dict(torch.load(r"model/model_7.pt"))
model.load_state_dict(torch.load(r"model\model_1\checkpoint_12.pt"))
model.eval()

dummy_input = torch.zeros(512, dtype=torch.int)
print(type(dummy_input))
input()
torch.onnx.export(model,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  r"model/model_1_12.onnx",   # where to save the model (can be a file or file-like object)
                  verbose=True
                  )
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules import TransformerEncoderLayer

import numpy as np

# Define the size of the character set
CHARSET_SIZE = 26  # A-Z
class VerboseTransformerEncoderLayer(TransformerEncoderLayer):
    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, w_x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)
        
        self.attention_weights = w_x

        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class VerboseTransformerModel(nn.Module):

    def __init__(self, nhead, adim, nhid, nlayers, dropout=0.5, output_att=False):
        super(VerboseTransformerModel, self).__init__()
        self.adim = adim
        self.output_att = output_att
        # encoder_layers = nn.TransformerEncoderLayer(adim, nhead, nhid, dropout, batch_first = True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder = VerboseTransformerEncoderLayer(adim, nhead, nhid, dropout, batch_first = True)

        self.classifier = nn.Linear(nhid*adim, 1)

    def forward(self, src):
        src = F.one_hot(src.to(torch.int64), CHARSET_SIZE).float()
        src = self.resize_embedding(src)
        src = self.transformer_encoder(src)
        src = torch.flatten(src, -2)
        
        # don't use a sigmoid function on the output inside the mode
        src = self.classifier(src)
        if self.output_att:
            return src, self.transformer_encoder.attention_weights
        return src

    def resize_embedding(self, input_tensor):
        input_size = list(input_tensor.shape)
        input_size[-1] = self.adim - CHARSET_SIZE
        zeros_tensor = torch.zeros(input_size, dtype=input_tensor.dtype, device=input_tensor.device)
        output_tensor = torch.cat([input_tensor, zeros_tensor], dim=-1)

        return output_tensor


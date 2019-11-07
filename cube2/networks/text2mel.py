#
# Author: Tiberiu Boros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import sys
import torch.nn as nn
import numpy as np

sys.path.append('')
from cube2.networks.modules import Attention


class Text2Mel(nn.Module):
    def __init__(self, encodings, char_emb_size=100, encoder_size=200, encoder_layers=1, decoder_size=300,
                 decoder_layers=1, mgc_size=80, pframes=3):
        super(Text2Mel, self).__init__()

        self.encodings = encodings
        self.pframes = pframes
        self.char_emb = nn.Embedding(len(self.encodings.char2int), char_emb_size)
        self.char_conv = nn.Sequential(nn.Conv1d(char_emb_size, char_emb_size, 5, padding=2), nn.ReLU(),
                                       nn.Dropout(0.33))

        self.encoder = nn.LSTM(char_emb_size, encoder_size, encoder_layers, bias=True,
                               dropout=0 if encoder_layers == 1 else 0.33, bidirectional=True)
        self.decoder = nn.LSTM(encoder_size * 2 + mgc_size, decoder_size, decoder_layers, bias=True,
                               dropout=0 if decoder_layers == 1 else 0.33,
                               bidirectional=False)
        self.dropout = nn.Dropout(0.33)
        self.output_mgc = nn.Sequential(nn.Linear(decoder_size, mgc_size * pframes), nn.Sigmoid())
        self.output_stop = nn.Sequential(nn.Linear(decoder_size, 1), nn.Tanh())
        self.att = Attention(encoder_size, decoder_size)

    def forward(self, input, gs_mgc=None):
        if gs_mgc is not None:
            tmp = []
            for ii in range(gs_mgc.shape[1] // self.pframes):
                index = ii * self.pframes + self.pframes - 1
                if index < gs_mgc.shape[1]:
                    tmp.append(gs_mgc[:, index, :])
                else:
                    tmp.append(torch.zeros((gs_mgc.shape[0], 1, gs_mgc.shape[2])))

            gs_mgc = torch.cat(tmp, dim=1)

        while True:
            pass

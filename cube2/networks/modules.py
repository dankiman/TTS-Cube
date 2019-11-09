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
import torch.nn as nn
import torch.functional as F


class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_scales):
        super(UpsampleNet, self).__init__()
        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))
        self.output_transform = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            noise = noise * 0.01
            # noise += 1
            x = x + noise
        x = torch.clamp(x, min=0, max=1)
        c = x.permute(0, 2, 1)
        if self.upsample_conv is not None:
            # B x 1 x C x T'
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
        return self.output_transform(c.permute(0, 2, 1))


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.softmax(torch.bmm(v, energy).squeeze(1), dim=1)

        a = attention.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.squeeze(1)

        return attention, weighted

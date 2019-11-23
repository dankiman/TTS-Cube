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
import json

sys.path.append('')
from cube2.networks.modules import Attention, PostNet
from os.path import exists


class Text2Mel(nn.Module):
    def __init__(self, encodings, char_emb_size=100, encoder_size=256, encoder_layers=1, decoder_size=1024,
                 decoder_layers=2, mgc_size=80, pframes=3):
        super(Text2Mel, self).__init__()
        self.MGC_PROJ_SIZE = 256

        self.encodings = encodings
        self.pframes = pframes
        self.mgc_order = mgc_size
        self.char_emb = nn.Embedding(len(self.encodings.char2int), char_emb_size, padding_idx=0)
        self.case_emb = nn.Embedding(4, 16, padding_idx=0)
        self.char_conv = nn.Sequential(nn.Conv1d(char_emb_size + 16, 512, 5, padding=2), nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 5, padding=2), nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 5, padding=2), nn.ReLU(),
                                       nn.Dropout(0.5)
                                       )
        self.mgc_proj = nn.Sequential(nn.Linear(mgc_size, self.MGC_PROJ_SIZE), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(self.MGC_PROJ_SIZE, self.MGC_PROJ_SIZE), nn.ReLU(), nn.Dropout(0.5))
        self.encoder = nn.LSTM(512, encoder_size, encoder_layers, bias=True,
                               dropout=0 if encoder_layers == 1 else 0.33, bidirectional=True)
        self.decoder = nn.LSTM(encoder_size * 2 + self.MGC_PROJ_SIZE, decoder_size, decoder_layers, bias=True,
                               dropout=0 if decoder_layers == 1 else 0.33,
                               bidirectional=False)

        self.dec2hid = nn.Sequential(nn.Linear(decoder_size, 500), nn.ReLU(), nn.Dropout(0.5))
        self.dropout = nn.Dropout(0.33)
        self.output_mgc = nn.Sequential(nn.Linear(500, mgc_size * pframes))
        self.output_stop = nn.Sequential(nn.Linear(mgc_size * pframes, self.pframes), nn.Sigmoid())
        self.att = Attention(encoder_size, decoder_size)
        self.postnet = PostNet(mgc_size)

    def forward(self, input, gs_mgc=None):
        if gs_mgc is not None:
            max_len = max([mgc.shape[0] for mgc in gs_mgc])
            # gs_mgc = torch.tensor(gs_mgc, dtype=self._get_device())
            tmp = np.zeros((len(gs_mgc), max_len // self.pframes, self.mgc_order))
            for iFrame in range(max_len // self.pframes):
                index = iFrame * self.pframes + self.pframes - 1
                for iB in range(len(gs_mgc)):
                    if index < gs_mgc[iB].shape[0]:
                        for zz in range(self.mgc_order):
                            tmp[iB, iFrame, zz] = gs_mgc[iB][index, zz]

            gs_mgc = torch.tensor(tmp, device=self._get_device(), dtype=torch.float)
        index = 0
        # input
        x = self._make_input(input)
        lstm_input = self.char_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_output, encoder_hidden = self.encoder(lstm_input.permute(1, 0, 2))
        encoder_output = encoder_output.permute(1, 0, 2)

        _, decoder_hidden = self.decoder(
            torch.zeros((1, encoder_output.shape[0], encoder_output.shape[2] + self.MGC_PROJ_SIZE),
                        device=self._get_device()))
        last_mgc = torch.zeros((lstm_input.shape[0], self.mgc_order), device=self._get_device())
        lst_output = []
        lst_stop = []
        lst_att = []

        while True:
            att_vec, att = self.att(decoder_hidden[-1][-1].unsqueeze(0), encoder_output)
            lst_att.append(att_vec.unsqueeze(1))
            m_proj = self.mgc_proj(last_mgc)
            # if gs_mgc is None:
            #    m_proj = torch.dropout(m_proj, 0.5, True)

            decoder_input = torch.cat((att, m_proj), dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), hx=decoder_hidden)
            decoder_output = decoder_output.permute(1, 0, 2)
            decoder_output = self.dec2hid(decoder_output)
            out_mgc = self.output_mgc(decoder_output)
            out_stop = self.output_stop(out_mgc.detach())
            for iFrame in range(self.pframes):
                lst_output.append(out_mgc[:, :, iFrame * self.mgc_order:iFrame * self.mgc_order + self.mgc_order])
                lst_stop.append(out_stop[:, :, iFrame])
            if gs_mgc is not None:
                last_mgc = gs_mgc[:, index, :]
            else:
                last_mgc = out_mgc[:, :, -self.mgc_order:].squeeze(1)
            index += 1
            if gs_mgc is not None and index == gs_mgc.shape[1]:
                break
            elif gs_mgc is None:
                if any(out_stop[0][-1].detach().cpu().numpy() > 0.5):
                    break
                # failsafe
                if index > x.shape[1] * 5:
                    break
        mgc = torch.cat(lst_output, dim=1)  # .view(len(input), -1, self.mgc_order)
        stop = torch.cat(lst_stop, dim=1)  # .view(len(input), -1)
        att = torch.cat(lst_att, dim=1)
        return mgc + self.postnet(mgc), mgc, stop, att

    def _make_input(self, input):
        max_len = max([len(seq) for seq in input])
        x_char = np.zeros((len(input), max_len), dtype=np.int32)
        x_case = np.zeros((len(input), max_len), dtype=np.int32)
        for iBatch in range(x_char.shape[0]):
            for iToken in range(x_char.shape[1]):
                if iToken < len(input[iBatch]):
                    char = input[iBatch][iToken]
                    case = 0
                    if char.lower() == char.upper():
                        case = 1  # symbol
                    elif char.lower() != char:
                        case = 2  # upper
                    else:
                        case = 3  # lower
                    char = char.lower()
                    if char in self.encodings.char2int:
                        char = self.encodings.char2int[char]
                    else:
                        char = 1  # UNK
                    x_char[iBatch, iToken] = char
                    x_case[iBatch, iToken] = case

        x_char = torch.tensor(x_char, device=self._get_device(), dtype=torch.long)
        x_case = torch.tensor(x_case, device=self._get_device(), dtype=torch.long)
        x_char = self.char_emb(x_char)
        x_case = self.case_emb(x_case)
        return torch.cat((x_char, x_case), dim=2)

    def _get_device(self):
        if self.case_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.case_emb.weight.device.type, str(self.case_emb.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


class DataLoader:
    def __init__(self, dataset):
        from cube2.io_modules.dataset import DatasetIO
        self._dio = DatasetIO()
        self._dataset = dataset
        self._file_index = 0
        self._frame_index = 0
        self._cur_x = []
        self._cur_mgc = []

    def _read_next(self):
        if self._file_index == len(self._dataset.files):
            self._file_index = 0
        file = self._dataset.files[self._file_index]
        mgc_file = file + ".mgc.npy"
        mgc = np.load(mgc_file)
        txt_file = file + ".txt"
        lab_file = file + ".lab"
        if exists(lab_file):
            json_obj = json.load(open(lab_file))
            trans = json_obj['transcription']
        else:
            txt = open(txt_file).read().strip()
            trans = [c for c in txt]
        self._file_index += 1
        return trans, mgc

    def get_batch(self, batch_size, mini_batch_size=16, device='cuda:0'):
        batch_mgc = []
        batch_x = []
        while len(batch_x) < batch_size:
            x, mgc = self._read_next()
            batch_x.append(x)
            batch_mgc.append(mgc)
        return batch_x, batch_mgc
        # return torch.tensor(batch_x, device=device, dtype=torch.float32), \
        #       torch.tensor(batch_mgc, device=device, dtype=torch.float32)


def _eval(model, dataset, params):
    return 0


def _update_encodings(encodings, dataset):
    import tqdm
    for train_file in tqdm.tqdm(dataset._dataset.files):
        txt_file = train_file + ".txt"
        lab_file = train_file + ".lab"
        if exists(lab_file):
            json_obj = json.load(open(lab_file))
            trans = json_obj['transcription']
        else:
            txt = open(txt_file).read().strip()
            trans = [c for c in trans]
        for char in trans:
            from cube2.io_modules.dataset import PhoneInfo
            pi = PhoneInfo(char, [], -1, -1)
            encodings.update(pi)


def _make_batch(gs_mgc, device='cpu'):
    max_len = max([mgc.shape[0] for mgc in gs_mgc])
    if max_len % 3 != 0:
        max_len = (max_len // 3) * 3
    # gs_mgc = torch.tensor(gs_mgc, dtype=self._get_device())
    tmp_mgc = np.zeros((len(gs_mgc), max_len, gs_mgc[0].shape[1]))
    tmp_stop = np.zeros((len(gs_mgc), max_len))
    for ii in range(max_len):
        index = ii
        for iB in range(len(gs_mgc)):
            if index < gs_mgc[iB].shape[0]:
                tmp_stop[iB, ii] = 0.0
                for zz in range(tmp_mgc.shape[2]):
                    tmp_mgc[iB, ii, zz] = gs_mgc[iB][index, zz]
            else:
                tmp_stop[iB, ii] == 1.0

    gs_mgc = torch.tensor(tmp_mgc, device=device, dtype=torch.float)
    gs_stop = torch.tensor(tmp_stop, device=device, dtype=torch.float)
    return gs_mgc, gs_stop


def _compute_guided_attention(num_tokens, num_mgc, device='cpu'):
    max_num_toks = max(num_tokens)
    max_num_mgc = max(num_mgc)
    target_probs = np.zeros((len(num_tokens), max_num_mgc, max_num_toks))

    for iBatch in range(len(num_tokens)):
        for iDecStep in range(max_num_mgc):
            for iAttIndex in range(max_num_toks):
                cChars = num_tokens[iBatch]
                cDecSteps = num_mgc[iBatch]
                t1 = iDecStep / cDecSteps
                value = 1.0 - np.exp(-((float(iAttIndex) / cChars - t1) ** 2) / 0.1)
                target_probs[iBatch, iDecStep, iAttIndex] = value

    return torch.tensor(target_probs, device=device)


def _start_train(params):
    from cube2.io_modules.dataset import Dataset, Encodings
    import tqdm

    trainset = Dataset("data/processed/train")
    devset = Dataset("data/processed/dev")
    sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
        len(devset.files)) + ' development files\n')
    epoch = 1
    patience_left = params.patience
    trainset = DataLoader(trainset)
    devset = DataLoader(devset)
    encodings = Encodings()
    if params.resume:
        encodings.load('data/text2mel.encodings')
    else:
        _update_encodings(encodings, trainset)
        encodings.store('data/text2mel.encodings')
    text2mel = Text2Mel(encodings)
    if params.resume:
        text2mel.load('data/text2mel.last')
    text2mel.to('cuda:0')
    optimizer_gen = torch.optim.Adam(text2mel.parameters(), lr=params.lr)
    text2mel.save('data/text2mel.last')

    test_steps = 500
    global_step = 0
    best_gloss = _eval(text2mel, devset, params)
    bce_loss = torch.nn.BCELoss()
    abs_loss = torch.nn.L1Loss(reduction='mean')
    mse_loss = torch.nn.MSELoss(reduction='mean')
    while patience_left > 0:
        text2mel.train()
        total_loss_bce = 0.0
        progress = tqdm.tqdm(range(test_steps))
        for step in progress:
            sys.stdout.flush()
            sys.stderr.flush()
            global_step += 1
            x, mgc = trainset.get_batch(batch_size=params.batch_size)
            pred_mgc, pred_pre, pred_stop, pred_att = text2mel(x, gs_mgc=mgc)
            target_mgc, target_stop = _make_batch(mgc, device=params.device)

            num_tokens = [len(seq) for seq in x]
            num_mgcs = [m.shape[0] // 3 for m in mgc]
            if not params.disable_guided_attention:
                target_att = _compute_guided_attention(num_tokens, num_mgcs, device=params.device)
            loss_bce = abs_loss(pred_mgc.view(-1), target_mgc.view(-1)) * 80 + \
                       abs_loss(pred_pre.view(-1), target_mgc.view(-1)) * 80 + \
                       bce_loss(pred_stop.view(-1), target_stop.view(-1))  # + \
            if not params.disable_guided_attention:
                loss_bce += (pred_att * target_att).mean()
            loss = loss_bce
            optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text2mel.parameters(), 5.)
            optimizer_gen.step()
            lss_bce = loss_bce.item()
            total_loss_bce += lss_bce

            progress.set_description('BCE_LOSS={0:.4}'.format(lss_bce))

        g_loss = _eval(text2mel, devset, params)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.write(
            '\tGlobal step {0} BCE_LOSS={1:.4}\n'.format(global_step, total_loss_bce / test_steps))
        sys.stdout.write('\tDevset evaluation: {0}\n'.format(g_loss))
        if g_loss < best_gloss:
            best_gloss = g_loss
            sys.stdout.write('\tStoring data/text2mel.best\n')
            text2mel.save('data/text2mel.best')

        if not np.isnan(total_loss_bce):
            sys.stdout.write('\tStoring data/text2mel.last\n')
            text2mel.save('data/text2mel.last')
        else:
            sys.stdout.write('exiting because of nan loss')
            sys.exit(0)


def _test_synth(params):
    from cube2.io_modules.dataset import Dataset, Encodings

    encodings = Encodings()
    encodings.load('data/text2mel.encodings')
    text2mel = Text2Mel(encodings)
    text2mel.load('data/text2mel.last')
    text2mel.to(params.device)
    text2mel.eval()
    mgc, stop, att = text2mel(['This is a simple test'])
    # from ipdb import set_trace
    # set_trace()
    mgc = mgc[0].detach().cpu().numpy()
    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = mgc[x, y]
            color = np.clip(val * 255, 0, 255)
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]  # bitmap[y, x] = [color, color, color]
    from PIL import Image
    img = Image.fromarray(bitmap)
    img.save('test.png')

    att = att[0].detach().cpu().numpy()
    new_att = np.zeros((att.shape[1], att.shape[0], 3), dtype=np.uint8)
    for ii in range(att.shape[1]):
        for jj in range(att.shape[0]):
            val = np.clip(int(att[jj, ii] * 255), 0, 255)
            new_att[ii, jj, 0] = val
            new_att[ii, jj, 1] = val
            new_att[ii, jj, 2] = val

    img = Image.fromarray(new_att)
    img.save('test.att.png')


if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('--patience', action='store', dest='patience', default=20, type='int',
                      help='Num epochs without improvement (default=20)')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='16', type='int',
                      help='number of samples in a single batch (default=32)')
    parser.add_option("--resume", action='store_true', dest='resume', help='Resume from previous checkpoint')
    parser.add_option("--use-gan", action='store_true', dest='use_gan', help='Resume from previous checkpoint')
    parser.add_option("--synth-test", action="store_true", dest="test")
    parser.add_option("--temperature", action="store", dest="temperature", type='float', default=1.0)
    parser.add_option("--device", action="store", dest="device", default='cuda:0')
    parser.add_option("--lr", action="store", dest="lr", default=1e-3, type=float)
    parser.add_option("--disable-guided-attention", action="store_true", dest="disable_guided_attention")

    (params, _) = parser.parse_args(sys.argv)
    if params.test:
        _test_synth(params)
    else:
        _start_train(params)

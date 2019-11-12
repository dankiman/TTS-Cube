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

import optparse
import torch
import sys
import torch.nn as nn
import numpy as np
import tqdm

sys.path.append('')
from cube2.io_modules.dataset import Dataset
from cube2.networks.vocoder import CubeNet


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

        if self._frame_index >= len(self._cur_mgc) - 1:
            self._file_index += 1
            self._frame_index = 0

            if self._file_index == len(self._dataset.files):
                self._file_index = 0
            file = self._dataset.files[self._file_index]
            mgc_file = file + ".mgc.npy"
            self._cur_mgc = np.load(mgc_file)
            wav_file = file + ".orig.wav"
            data, sample_rate = self._dio.read_wave(wav_file)
            self._cur_x = data

        result = [self._cur_mgc[self._frame_index], self._cur_x[self._frame_index * 256:self._frame_index * 256 + 256]]

        self._frame_index += 1
        return result

    def get_batch(self, batch_size, mini_batch_size=16, device='cuda:0'):
        batch_mgc = []
        batch_x = []
        while len(batch_mgc) < batch_size:
            mini_batch_mgc = []
            mini_batch_x = []
            for ii in range(mini_batch_size):
                mgc, x = self._read_next()
                mini_batch_mgc.append(mgc)
                mini_batch_x.append(x)
            batch_x.append(mini_batch_x)
            batch_mgc.append(mini_batch_mgc)

        return torch.tensor(batch_x, device=device, dtype=torch.float32), \
               torch.tensor(batch_mgc, device=device, dtype=torch.float32)


def gaussian_loss(mean, logvar, y, log_std_min=-7.0):
    import math
    log_std = torch.clamp(logvar, min=log_std_min).view(-1)
    mean = mean.view(-1)
    y = y.view(-1)
    log_probs = -0.5 * (- math.log(2.0 * math.pi) - 2. * log_std - torch.pow(y - mean, 2) * torch.exp((-2.0 * log_std)))
    return log_probs.squeeze().mean()


def _eval(model, dataset, params):
    model.eval()
    lss_gauss = 0
    with torch.no_grad():
        for step in tqdm.tqdm(range(100)):
            x, mgc = dataset.get_batch(batch_size=params.batch_size)
            mean, logvar, pred_y = model(mgc, x=x)
            loss_gauss = gaussian_loss(mean, logvar, x)
            lss_gauss += loss_gauss.item()
    return lss_gauss / 100


def _start_train(params):
    from cube2.io_modules.dataset import Dataset

    trainset = Dataset("data/processed/train")
    devset = Dataset("data/processed/dev")
    sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
        len(devset.files)) + ' development files\n')
    epoch = 1
    patience_left = params.patience
    trainset = DataLoader(trainset)
    devset = DataLoader(devset)
    cubenet = CubeNet()
    if params.resume:
        cubenet.load('data/cube.last')
    cubenet.to('cuda:0')
    optimizer_gen = torch.optim.Adam(cubenet.parameters(), lr=params.lr)

    test_steps = 500
    global_step = 0
    best_gloss = _eval(cubenet, devset, params)
    while patience_left > 0:
        cubenet.train()
        total_loss_gauss = 0.0
        progress = tqdm.tqdm(range(test_steps))
        for step in progress:
            sys.stdout.flush()
            sys.stderr.flush()
            global_step += 1
            x, mgc = trainset.get_batch(batch_size=params.batch_size)
            # from ipdb import set_trace
            # set_trace()
            mean, logvar, pred_y = cubenet(mgc, x=x)

            loss_gauss = gaussian_loss(mean, logvar, x)
            loss = loss_gauss
            optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cubenet.parameters(), 5.)
            optimizer_gen.step()
            lss_gauss = loss_gauss.item()
            total_loss_gauss += lss_gauss

            progress.set_description('GAUSSIAN_LOSS={0:.4}'.format(lss_gauss))

        g_loss = _eval(cubenet, devset, params)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.write(
            '\tGlobal step {0} GAUSSIAN_LOSS={1:.4}\n'.format(global_step, total_loss_gauss / test_steps))
        sys.stdout.write('\tDevset evaluation: {0}\n'.format(g_loss))
        if g_loss < best_gloss:
            best_gloss = g_loss
            sys.stdout.write('\tStoring data/cube.best\n')
            cubenet.save('data/cube.best')

        if not np.isnan(total_loss_gauss):
            sys.stdout.write('\tStoring data/cube.last\n')
            cubenet.save('data/cube.last')
        else:
            sys.stdout.write('exiting because of nan loss')
            sys.exit(0)


def _test_synth(params):
    devset = Dataset("data/processed/dev")
    devset = DataLoader(devset)
    cubenet = CubeNet()
    cubenet.load('data/cube.last')
    cubenet.to(params.device)
    cubenet.eval()
    import time
    x, mgc = devset.get_batch(batch_size=params.batch_size, device=params.device, mini_batch_size=64)
    start = time.time()
    with torch.no_grad():
        mean, logvar, pred_y = cubenet(mgc, temperature=params.temperature, eps_min=-12)
    end = time.time()
    synth = torch.clamp(pred_y.view(-1) * 32767, min=-32767, max=32767)
    from cube2.io_modules.dataset import DatasetIO
    dio = DatasetIO()
    dio.write_wave('gan.wav', synth.detach().cpu().numpy(), 16000, dtype=np.int16)
    synth = x.view(-1) * 32767
    dio.write_wave('orig.wav', synth.detach().cpu().numpy(), 16000, dtype=np.int16)
    sys.stdout.write(
        'Actual execution time took {0} for {1} seconds of audio.\n'.format(end - start, len(x.view(-1)) / 16000))


if __name__ == '__main__':
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

    (params, _) = parser.parse_args(sys.argv)
    if params.test:
        _test_synth(params)
    else:
        _start_train(params)

import optparse
import torch
import sys
import torch.nn as nn
import numpy as np

sys.path.append('')
from torch.distributions.normal import Normal
from cube.models.clarinet.wavenet import Wavenet


class CubeNet(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, lstm_layers=1, output_size=128, ext_conditioning_size=0,
                 upsample_scales=[2]):
        super(CubeNet, self).__init__()
        self._output_size = output_size

        self._upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self._upsample_conv.append(convt)
            self._upsample_conv.append(nn.LeakyReLU(0.4))

        self._lstm = nn.LSTM(mgc_size + output_size + ext_conditioning_size, lstm_size, bidirectional=False,
                             num_layers=lstm_layers)
        self._output_mean = nn.Linear(lstm_size, output_size)
        self._output_logvar = nn.Linear(lstm_size, output_size)

    def forward(self, mgc, ext_conditioning=None, temperature=1.0, gs_x=None):
        zeros = np.zeros((mgc.shape[0], mgc.shape[1], self._output_size))
        ones = np.ones((mgc.shape[0], mgc.shape[1], self._output_size))

        q_0 = Normal(torch.tensor(zeros, dtype=torch.float32).to(mgc.device.type),
                     torch.tensor(ones, dtype=torch.float32).to(mgc.device.type))
        z = q_0.sample() * temperature
        from ipdb import set_trace
        set_trace()
        mgc = self.upsample(mgc)

    def upsample(self, c):
        c = c.permute(0, 2, 1)

        if self._upsample_conv is not None:
            # B x 1 x C x T'
            c = c.unsqueeze(1)
            for f in self._upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
        c = c.permute(0, 2, 1)
        return c


class DiscriminatorWavenet(nn.Module):
    def __init__(self, mgc_size=80):
        super(DiscriminatorWavenet, self).__init__()
        self._model = Wavenet(kernel_size=2,
                              num_layers=11,
                              residual_channels=128,
                              gate_channels=256,
                              skip_channels=128,
                              cin_channels=mgc_size,
                              out_channels=1,
                              upsample_scales=[16, 16])

    def forward(self, signal, mgc):
        return torch.sigmoid(self._model(signal, mgc))


class DataLoader:
    def __init__(self, dataset):
        from cube.io_modules.dataset import DatasetIO
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

    def get_batch(self, batch_size, device='cpu'):
        batch_mgc = []
        batch_x = []
        while len(batch_mgc) < batch_size:
            mini_batch_mgc = []
            mini_batch_x = []
            for ii in range(8):
                mgc, x = self._read_next()
                mini_batch_mgc.append(mgc)
                mini_batch_x.append(x)
            batch_x.append(mini_batch_x)
            batch_mgc.append(mini_batch_mgc)

        return torch.tensor(batch_x, device=device, dtype=torch.float32), \
               torch.tensor(batch_mgc, device=device, dtype=torch.float32)


def _start_train(params):
    from cube.io_modules.dataset import Dataset

    trainset = Dataset("data/processed/train")
    devset = Dataset("data/processed/dev")
    sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
        len(devset.files)) + ' development files\n')
    epoch = 1
    patience_left = params.patience
    trainset = DataLoader(trainset)
    devset = DataLoader(devset)
    cubenet = CubeNet()
    discnet = DiscriminatorWavenet()
    optimizer_gen = torch.optim.Adam(cubenet.parameters())
    optimizer_dis = torch.optim.Adam(discnet.parameters())

    while patience_left > 0:
        x, mgc = trainset.get_batch(batch_size=params.batch_size)
        output = cubenet(mgc, gs_x=x)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--patience', action='store', dest='patience', default=20, type='int',
                      help='Num epochs without improvement (default=20)')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='32', type='int',
                      help='number of samples in a single batch (default=32)')
    parser.add_option("--resume", action='store_true', dest='resume', help='Resume from previous checkpoint')

    (params, _) = parser.parse_args(sys.argv)
    _start_train(params)

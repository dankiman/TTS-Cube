import optparse
import torch
import sys
import torch.nn as nn
import numpy as np

sys.path.append('')
from torch.distributions.normal import Normal
from cube.models.clarinet.wavenet import Wavenet


class CubeNet(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, lstm_layers=1, output_size=128, ext_conditioning_size=0):
        super(CubeNet, self).__init__()
        self._output_size = output_size

        self._lstm = nn.LSTM(mgc_size + output_size + ext_conditioning_size, lstm_size, bidirectional=False,
                             num_layers=lstm_layers)
        self._output_mean = nn.Linear(lstm_size, output_size)
        self._output_logvar = nn.Linear(lstm_size, output_size)

    def forward(self, mgc, ext_conditioning=None, temperature=1.0):
        zeros = np.zeros((mgc.shape[0], mgc.shape[1], self._output_size))
        ones = np.ones((mgc.shape[0], mgc.shape[1], self._output_size))

        q_0 = Normal(torch.tensor(zeros, dtype=torch.float32).to(mgc.device.type),
                     torch.tensor(ones, dtype=torch.float32).to(mgc.device.type))
        z = q_0.sample() * temperature


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
        self._dataset = dataset
        self._file_index = 0
        self._frame_index = 0

    def get_batch(self, batch_size, repeat_mgc=2, device='cuda:0'):
        mgc_batch = []
        x_batch = []
        disc_mgc_batch = []

        return torch.tensor(x_batch, device=device), \
               torch.tensor(mgc_batch, device=device), \
               torch.tensor(disc_mgc_batch, device=device)


def _start_train(params):
    from cube.io_modules.dataset import Dataset
    from cube.io_modules.dataset import DatasetIO
    trainset = Dataset("data/processed/train")
    devset = Dataset("data/processed/dev")
    sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
        len(devset.files)) + ' development files\n')
    epoch = 1
    left_itt = params.patience
    dio = DatasetIO()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--patience', action='store', dest='patience',
                      help='Num epochs without improvement (default=20)')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='32', type='int',
                      help='number of samples in a single batch (default=32)')
    parser.add_option("--resume", action='store_true', dest='resume', help='Resume from previous checkpoint')

    (params, _) = parser.parse_args(sys.argv)
    _start_train(params)

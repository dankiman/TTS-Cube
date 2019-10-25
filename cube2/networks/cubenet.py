import optparse
import torch
import sys
import torch.nn as nn
import numpy as np
import tqdm

sys.path.append('')
from torch.distributions.normal import Normal
from cube.models.clarinet.wavenet import Wavenet
from cube.io_modules.dataset import Dataset


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
        c = x.permute(0, 2, 1)
        if self.upsample_conv is not None:
            # B x 1 x C x T'
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
        return self.output_transform(c.permute(0, 2, 1))


class CubeNet(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, lstm_layers=1, upsample_scales=[4, 4, 4, 4]):
        super(CubeNet, self).__init__()
        self.upsample = UpsampleNet(mgc_size, 256, upsample_scales)
        self.rnn = nn.LSTM(256 + 1, lstm_size, num_layers=lstm_layers)
        self.output = nn.Linear(lstm_size, 2)

    def forward(self, mgc, ext_conditioning=None, temperature=1.0, x=None):
        cond = self.upsample(mgc)
        if x is not None:
            x = x.view(x.shape[0], -1)
            x = torch.cat((torch.zeros((x.shape[0], 1), device=x.device.type), x[:, 0:-1]), dim=1)
            rnn_input = torch.cat((cond, x.unsqueeze(2)), dim=2)
            rnn_output, _ = self.rnn(rnn_input.permute(1, 0, 2))
            rnn_output = rnn_output.permute(1, 0, 2)
            output = self.output(rnn_output)
            mean = torch.tanh(output[:, :, 0])
            logvar = output[:, :, 1]
            eps = torch.randn_like(mean)
            zz = self._reparameterize(mean, logvar, eps * temperature)
        else:
            mean_list = []
            logvar_list = []
            zz_list = []
            hidden = None
            x = torch.zeros((mgc.shape[0], 1), device=mgc.device.type)
            for ii in range(cond.shape[1]):
                rnn_input = torch.cat((cond[:, ii, :].unsqueeze(1), x.unsqueeze(2)), dim=2)
                rnn_output, hidden = self.rnn(rnn_input.permute(1, 0, 2), hx=hidden)
                rnn_output = rnn_output.permute(1, 0, 2)
                output = self.output(rnn_output)
                mean = torch.tanh(output[:, :, 0])
                logvar = output[:, :, 1]
                eps = torch.randn_like(mean)
                zz = self._reparameterize(mean, logvar, eps * temperature)
                mean_list.append(mean)
                logvar_list.append(logvar)
                zz_list.append(zz)
                x = zz

            mean = torch.cat(mean_list, dim=1)
            logvar = torch.cat(logvar_list, dim=1)
            zz = torch.cat(zz_list, dim=1)

        return mean, logvar, zz

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _reparameterize(self, mu, logvar, eps):
        logvar = torch.clamp(logvar, min=-7)
        std = torch.exp(logvar)
        return mu + eps * std


class DiscriminatorWavenet(nn.Module):
    def __init__(self, mgc_size=80):
        super(DiscriminatorWavenet, self).__init__()

    def forward(self, signal, mgc):
        pass  # return self._model(signal, mgc)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


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

    def get_batch(self, batch_size, device='cuda:0'):
        batch_mgc = []
        batch_x = []
        while len(batch_mgc) < batch_size:
            mini_batch_mgc = []
            mini_batch_x = []
            for ii in range(16):
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
    if params.resume:
        cubenet.load('data/cube.last')
    cubenet.to('cuda:0')
    optimizer_gen = torch.optim.Adam(cubenet.parameters(), lr=1e-3)

    test_steps = 500
    global_step = 0

    while patience_left > 0:
        cubenet.train()
        total_loss_gauss = 0.0
        progress = tqdm.tqdm(range(test_steps))
        for step in progress:
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
        sys.stdout.write(
            'Global step {0} GAUSSIAN_LOSS={1:.4}\n'.format(global_step, total_loss_gauss / test_steps))

        cubenet.save('data/cube.last')


def _test_synth(params):
    devset = Dataset("data/processed/dev")
    devset = DataLoader(devset)
    cubenet = CubeNet()
    cubenet.load('data/cube.last')
    cubenet.to('cuda:0')
    cubenet.eval()
    import time
    x, mgc = devset.get_batch(batch_size=params.batch_size)
    start = time.time()
    with torch.no_grad():
        mean, logvar, pred_y = cubenet(mgc, temperature=params.temperature)
    end = time.time()
    synth = pred_y.view(-1) * 32000
    from cube.io_modules.dataset import DatasetIO
    dio = DatasetIO()
    dio.write_wave('gan.wav', synth.detach().cpu().numpy(), 16000, dtype=np.int16)
    synth = x.view(-1) * 32000
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

    (params, _) = parser.parse_args(sys.argv)
    if params.test:
        _test_synth(params)
    else:
        _start_train(params)

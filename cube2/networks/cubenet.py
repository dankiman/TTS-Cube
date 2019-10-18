import optparse
import torch
import sys
import torch.nn as nn
import numpy as np
import tqdm

sys.path.append('')
from torch.distributions.normal import Normal
from cube.models.clarinet.wavenet import Wavenet


class CubeNet(nn.Module):
    def __init__(self, mgc_size=80, lstm_size=500, lstm_layers=5, output_size=32, ext_conditioning_size=0,
                 upsample_scales=[2, 2, 2]):
        super(CubeNet, self).__init__()
        self._output_size = output_size

        self._upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self._upsample_conv.append(convt)
            self._upsample_conv.append(nn.LeakyReLU(0.4))

        lstm_list = [nn.LSTM(mgc_size + output_size * 2 + ext_conditioning_size, lstm_size, bidirectional=False,
                             num_layers=1) for _ in range(lstm_layers)]
        self._lstm_list = nn.ModuleList(lstm_list)
        self._output_mean = nn.ModuleList([nn.Linear(lstm_size, output_size) for _ in range(lstm_layers)])
        self._output_logvar = nn.ModuleList([nn.Linear(lstm_size, output_size) for _ in range(lstm_layers)])

    def forward(self, mgc, ext_conditioning=None, temperature=1.0, x=None):
        mgc = self._upsample(mgc)

        zeros = np.zeros((mgc.shape[0], mgc.shape[1], self._output_size))
        ones = np.ones((mgc.shape[0], mgc.shape[1], self._output_size))

        q_0 = Normal(torch.tensor(zeros, dtype=torch.float32).to(mgc.device.type),
                     torch.tensor(ones, dtype=torch.float32).to(mgc.device.type))
        z = q_0.sample() * temperature

        # we don't handle external conditioning yet
        x_list = []
        # x = torch.cat((torch.zeros(x.shape[0], 1, x.shape[2]), x), dim=1)
        x_list.append(torch.zeros(x.shape[0], 1, self._output_size).to('cuda:0'))
        for ii in range(x.shape[1]):
            for jj in range(x.shape[2] // self._output_size):
                x_list.append(x[:, ii, jj * self._output_size:jj * self._output_size + self._output_size].unsqueeze(1))
        x_list.pop(-1)
        x = torch.cat(x_list, dim=1)
        zz = z
        first = True
        for lstm, output_mean, output_logvar in zip(self._lstm_list, self._output_mean, self._output_logvar):
            lstm_input = torch.cat((x, mgc, zz), dim=2)
            output, hidden = lstm(lstm_input.permute(1, 0, 2))
            output = output.permute(1, 0, 2)
            mean = output_mean(output)
            logvar = output_logvar(output)
            if not first:
                zz = self._reparameterize(mean, logvar, zz) + zz
            else:
                zz = self._reparameterize(mean, logvar, zz)
                first = False

        return mean, logvar, zz

    def _upsample(self, c):
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

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _reparameterize(self, mu, logvar, eps):
        std = torch.exp(0.5 * logvar)
        return mu + eps * std


class DiscriminatorWavenet(nn.Module):
    def __init__(self, mgc_size=80):
        super(DiscriminatorWavenet, self).__init__()
        # self._model = Wavenet(kernel_size=2,
        #                       num_layers=11,
        #                       residual_channels=8,
        #                       gate_channels=16,
        #                       skip_channels=8,
        #                       cin_channels=mgc_size,
        #                       out_channels=1,
        #                       upsample_scales=[16, 16])
        self._model = Wavenet(kernel_size=2,
                              num_layers=11,
                              residual_channels=8,
                              gate_channels=16,
                              skip_channels=8,
                              cin_channels=mgc_size,
                              out_channels=1,
                              upsample_scales=[16, 16])

    def forward(self, signal, mgc):
        return self._model(signal, mgc)

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
            for ii in range(64):
                mgc, x = self._read_next()
                mini_batch_mgc.append(mgc)
                mini_batch_x.append(x)
            batch_x.append(mini_batch_x)
            batch_mgc.append(mini_batch_mgc)

        return torch.tensor(batch_x, device=device, dtype=torch.float32), \
               torch.tensor(batch_mgc, device=device, dtype=torch.float32)


from cube.models.clarinet.modules import STFT

cstft = STFT(filter_length=512, hop_length=128).to('cuda:0')


def stft(y, scale='linear'):
    D = cstft(y.unsqueeze(0).unsqueeze(0))
    # D = stft(y, n_fft=512, hop_length=128, win_length=512, window=torch.hamming_window(512).cuda())
    real = D[0].squeeze(0).unsqueeze(2)
    imag = D[1].squeeze(0).unsqueeze(2)
    D = torch.cat((real, imag), dim=2)
    D = torch.sqrt(D.pow(2).sum(-1) + 1e-10)
    # D = torch.sqrt(torch.clamp(D.pow(2).sum(-1), min=1e-10))
    if scale == 'linear':
        return D
    elif scale == 'log':
        S = 2 * torch.log(torch.clamp(D, 1e-10, float("inf")))
        return S
    else:
        pass


def gaussian_loss(mean, logvar, y, log_std_min=-7.0):
    import math
    # assert y_hat.dim() == 3
    # assert y_hat.size(1) == 2

    # (B x T x C)
    # y_hat = y_hat.transpose(1, 2)

    # #mean = y_hat[:, :, :1]
    log_std = torch.clamp(logvar, min=log_std_min).view(-1)
    mean = mean.view(-1)
    y = y.view(-1)
    # mean

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
    discnet = DiscriminatorWavenet()
    cubenet.load('data/cube.last')
    cubenet.to('cuda:0')
    discnet.to('cuda:0')
    optimizer_gen = torch.optim.Adam(cubenet.parameters(), lr=1e-4)
    optimizer_dis = torch.optim.Adam(discnet.parameters(), lr=1e-4)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()
    test_steps = 500
    global_step = 0

    while patience_left > 0:
        total_loss_disc = 0.0
        total_loss_gen = 0.0
        total_loss_frame = 0.0
        total_loss_gauss = 0.0
        progress = tqdm.tqdm(range(test_steps))
        for step in progress:
            global_step += 1
            x, mgc = trainset.get_batch(batch_size=params.batch_size)
            from ipdb import set_trace
            set_trace()
            mean, logvar, pred_y = cubenet(mgc, x=x)
            # outputs_real = discnet(x.view(x.shape[0], -1).unsqueeze(1), mgc.permute(0, 2, 1))
            # outputs_synt = discnet(pred_y.detach().view(x.shape[0], -1).unsqueeze(1), mgc.permute(0, 2, 1))
            # outputs_real = outputs_real[:, :, -1]
            # outputs_synt = outputs_synt[:, :, -1]
            #
            # tgt_real = torch.ones(outputs_real.size()).to('cuda:0')
            # tgt_synt = torch.zeros(outputs_real.size()).to('cuda:0')
            # loss = bce_loss(outputs_real, tgt_real) + bce_loss(outputs_synt, tgt_synt)
            # total_loss_disc += loss.item()
            lss_disc = 0.0  # loss.item()
            # if lss_disc > 0.01:
            #     optimizer_dis.zero_grad()
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(discnet.parameters(), 1.)
            #     optimizer_dis.step()
            # outputs_synt = discnet(pred_y.view(x.shape[0], -1).unsqueeze(1), mgc.permute(0, 2, 1))
            # outputs_synt = outputs_synt[:, :, -1]
            # loss_bce = bce_loss(outputs_synt, tgt_real)

            fft_pred = stft(pred_y.view(-1), scale='log')
            fft_orig = stft(x.view(-1), scale='log')
            loss_mse = mse_loss(fft_pred.view(-1), fft_orig.view(-1))

            loss_gauss = gaussian_loss(mean, logvar, x)
            loss = loss_gauss + loss_mse  # + loss_bce  # loss_bce + loss_mse + loss_gauss
            optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cubenet.parameters(), 1.)
            optimizer_gen.step()
            total_loss_gen += 0.0  # loss_bce.item()
            total_loss_frame += loss_mse.item()
            lss_gen = 0.0  # loss_bce.item()
            lss_frm = loss_mse.item()
            lss_gauss = loss_gauss.item()
            total_loss_gauss += lss_gauss

            progress.set_description('D_LOSS={0:.4} G_LOSS={1:.4} F_LOSS={2:.4} N_LOSS={3:.4}'.format(lss_disc,
                                                                                                      lss_gen,
                                                                                                      lss_frm,
                                                                                                      lss_gauss))
        sys.stdout.write(
            'Global step {0} D_LOSS={1:.4} G_LOSS={2:.4} F_LOSS={3:.4} N_LOSS={4:.4}\n'.format(global_step,
                                                                                               total_loss_disc / test_steps,
                                                                                               total_loss_gen / test_steps,
                                                                                               total_loss_frame / test_steps,
                                                                                               total_loss_gauss / test_steps))

        discnet.save('data/disc.last')
        cubenet.save('data/cube.last')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--patience', action='store', dest='patience', default=20, type='int',
                      help='Num epochs without improvement (default=20)')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='16', type='int',
                      help='number of samples in a single batch (default=32)')
    parser.add_option("--resume", action='store_true', dest='resume', help='Resume from previous checkpoint')

    (params, _) = parser.parse_args(sys.argv)
    _start_train(params)

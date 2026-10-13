"""The decoder and its iSTFTNet vocoder: repo/kokoro/istftnet.py, statement for statement.

Arrays keep PyTorch's shapes, (batch, channels, time) unless a line says otherwise.
"""
import math

import mlx.core as mx
import numpy as np

from .nn import Conv1d, ConvTranspose1d, instance_norm, interpolate_linear, leaky_relu, linear, upsample_nearest


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def adain(x, s, w, name):
    """AdaIN1d (istftnet.py 20-31): instance norm, then (1 + gamma) * x + beta with gamma, beta = fc(s)."""
    h = linear(s, w, f"{name}.fc")                 # [B, 2C]
    C = h.shape[1] // 2
    gamma, beta = h[:, :C, None], h[:, C:, None]    # [B, C, 1]
    return (1 + gamma) * instance_norm(x) + beta


class AdaINResBlock1:
    """istftnet.py 34-77: three rounds of (AdaIN, Snake, dilated conv, AdaIN, Snake, conv) with a skip."""

    def __init__(self, w, name, kernel_size=3, dilation=(1, 3, 5)):
        self.w, self.name = w, name
        self.convs1 = [Conv1d(w, f"{name}.convs1.{i}", dilation=d, padding=get_padding(kernel_size, d)) for i, d in enumerate(dilation)]
        self.convs2 = [Conv1d(w, f"{name}.convs2.{i}", dilation=1, padding=get_padding(kernel_size, 1)) for i in range(len(dilation))]

    def __call__(self, x, s):
        w, n = self.w, self.name
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            a1, a2 = w[f"{n}.alpha1.{i}"], w[f"{n}.alpha2.{i}"]      # [1, C, 1]
            xt = adain(x, s, w, f"{n}.adain1.{i}")
            xt = xt + (1 / a1) * (mx.sin(a1 * xt) ** 2)             # Snake1D
            xt = c1(xt)
            xt = adain(xt, s, w, f"{n}.adain2.{i}")
            xt = xt + (1 / a2) * (mx.sin(a2 * xt) ** 2)             # Snake1D
            xt = c2(xt)
            x = xt + x
        return x


class STFT:
    """TorchSTFT (istftnet.py 80-105): torch.stft / torch.istft with a periodic Hann window, the signal
    centred and reflect-padded (torch's defaults), no normalization, one-sided spectrum."""

    def __init__(self, filter_length, hop_length, win_length):
        assert win_length == filter_length and filter_length % hop_length == 0
        self.n, self.hop = filter_length, hop_length
        k = np.arange(win_length, dtype=np.float64)
        win = (0.5 - 0.5 * np.cos(2.0 * np.pi * k / win_length)).astype(np.float32)   # torch.hann_window(periodic=True)
        self.window = mx.array(win)
        self.window_sq = (win.astype(np.float64) ** 2)

    def transform(self, x):
        """x [B, L] -> magnitude and phase, each [B, n/2 + 1, 1 + L // hop]."""
        n, hop = self.n, self.hop
        pad = n // 2
        x = mx.concatenate([x[:, pad:0:-1], x, x[:, -2:-pad - 2:-1]], axis=1)       # reflect, edge not repeated
        frames = 1 + (x.shape[1] - n) // hop
        idx = mx.arange(frames)[:, None] * hop + mx.arange(n)[None, :]
        spec = mx.fft.rfft(x[:, idx] * self.window, axis=-1)                         # [B, frames, n/2 + 1]
        real, imag = mx.real(spec), mx.imag(spec)
        magnitude = mx.sqrt(real * real + imag * imag)
        # torch.angle. A bin with no imaginary part and a negative real part has phase +pi. The first and
        # last bins are always such bins (they are sums of real numbers); on Apple GPUs MLX's FFT can
        # hand back -0.0 there, and atan2(-0.0, negative) is -pi, so the sign of zero is fixed first.
        phase = mx.arctan2(mx.where(imag == 0, mx.zeros_like(imag), imag), real)
        return magnitude.transpose(0, 2, 1), phase.transpose(0, 2, 1)

    def inverse(self, magnitude, phase):
        """magnitude, phase [B, n/2 + 1, frames] -> [B, 1, hop * (frames - 1)]."""
        n, hop = self.n, self.hop
        spec = (magnitude * mx.cos(phase)) + 1j * (magnitude * mx.sin(phase))        # magnitude * exp(1j * phase)
        frames = mx.fft.irfft(spec.transpose(0, 2, 1), n=n, axis=-1) * self.window   # [B, frames, n]
        B, F, _ = frames.shape
        # overlap-add: frame f starts at sample f * hop, so its k-th block of `hop` samples lands in block f + k
        parts = n // hop
        blocks = frames.reshape(B, F, parts, hop)
        out = sum(mx.pad(blocks[:, :, k, :], ((0, 0), (k, parts - 1 - k), (0, 0))) for k in range(parts))
        out = out.reshape(B, (F + parts - 1) * hop)
        # torch.istft divides by the overlap-added squared window, then drops the n/2 padding at both ends
        env = np.zeros((F + parts - 1) * hop, dtype=np.float64)
        for k in range(parts):
            env[k * hop:(k + F) * hop] += np.tile(self.window_sq[k * hop:(k + 1) * hop], F)
        start, length = n // 2, hop * (F - 1)
        out = out[:, start:start + length] / mx.array(env[start:start + length].astype(np.float32))
        return out[:, None, :]


class SineGen:
    """istftnet.py 108-209 with flag_for_pulse False: the harmonics of F0 as sine waves, plus noise."""

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        self.sine_amp, self.noise_std = sine_amp, noise_std
        self.harmonic_num, self.dim = harmonic_num, harmonic_num + 1
        self.sampling_rate, self.voiced_threshold, self.upsample_scale = samp_rate, voiced_threshold, upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def _f02sine(self, f0_values, rand_ini):
        """f0_values [B, L, dim]; rand_ini [B, dim] uniform in [0, 1)."""
        # F0 in cycles per sample; the whole number of cycles does not change the phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # random start phase for every harmonic but the fundamental
        rand_ini = mx.concatenate([mx.zeros_like(rand_ini[:, :1]), rand_ini[:, 1:]], axis=1)
        rad_values = mx.concatenate([rad_values[:, :1, :] + rand_ini[:, None, :], rad_values[:, 1:, :]], axis=1)
        # phase = 2 pi * running sum, taken at 1/upsample_scale of the rate and stretched back
        rad_values = interpolate_linear(rad_values.transpose(0, 2, 1), 1 / self.upsample_scale).transpose(0, 2, 1)
        phase = mx.cumsum(rad_values, axis=1) * 2 * math.pi
        phase = interpolate_linear(phase.transpose(0, 2, 1) * self.upsample_scale, self.upsample_scale).transpose(0, 2, 1)
        return mx.sin(phase)

    def __call__(self, f0, rand_ini, noise):
        """f0 [B, L, 1]; rand_ini [B, dim]; noise [B, L, dim] standard normal."""
        fn = f0 * mx.arange(1, self.harmonic_num + 2, dtype=mx.float32)[None, None, :]
        sine_waves = self._f02sine(fn, rand_ini) * self.sine_amp
        uv = self._f02uv(f0)
        # unvoiced: noise as loud as the sine would be; voiced: a little noise on top of the sine
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        return sine_waves * uv + noise_amp * noise, uv


class Generator:
    """istftnet.py 257-325."""

    def __init__(self, w, name, resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                 resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        self.w, self.name = w, name
        self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
        self.upsample_scale = math.prod(upsample_rates) * gen_istft_hop_size
        # SourceModuleHnNSF (istftnet.py 212-254): SineGen, then Linear + tanh to merge the harmonics
        self.sine_gen = SineGen(24000, self.upsample_scale, harmonic_num=8, voiced_threshold=10)
        self.ups = [ConvTranspose1d(w, f"{name}.ups.{i}", stride=u, padding=(k - u) // 2)
                    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))]
        self.resblocks, self.noise_convs, self.noise_res = [], [], []
        for i in range(len(self.ups)):
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(w, f"{name}.resblocks.{i * self.num_kernels + j}", k, d))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(w, f"{name}.noise_convs.{i}", stride=stride_f0, padding=(stride_f0 + 1) // 2))
                self.noise_res.append(AdaINResBlock1(w, f"{name}.noise_res.{i}", 7, [1, 3, 5]))
            else:
                self.noise_convs.append(Conv1d(w, f"{name}.noise_convs.{i}"))
                self.noise_res.append(AdaINResBlock1(w, f"{name}.noise_res.{i}", 11, [1, 3, 5]))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = Conv1d(w, f"{name}.conv_post", padding=3)
        self.stft = STFT(gen_istft_n_fft, gen_istft_hop_size, gen_istft_n_fft)

    def random_draws(self, n_f0_frames, batch=1):
        """The random numbers one call uses, in the order the official code draws them: the start phases
        (torch.rand, istftnet.py 150) and the source noise (torch.randn_like, 205). The official code also
        draws a third array (253) for a noise branch whose output is never used; it is skipped."""
        L = n_f0_frames * self.upsample_scale
        return mx.random.uniform(shape=(batch, self.sine_gen.dim)), mx.random.normal(shape=(batch, L, self.sine_gen.dim))

    def source(self, f0, draws=None):
        """f0 [B, frames] -> the harmonic source [B, L, 1] at audio rate."""
        f0 = upsample_nearest(f0[:, None, :], self.upsample_scale).transpose(0, 2, 1)      # bs, n, t
        rand_ini, noise = draws if draws is not None else self.random_draws(f0.shape[1] // self.upsample_scale, f0.shape[0])
        sine_wavs, _ = self.sine_gen(f0, rand_ini, noise)
        return mx.tanh(linear(sine_wavs, self.w, f"{self.name}.m_source.l_linear"))

    def __call__(self, x, s, f0, draws=None):
        har_source = self.source(f0, draws)[:, :, 0]                                       # [B, L]
        har_spec, har_phase = self.stft.transform(har_source)
        har = mx.concatenate([har_spec, har_phase], axis=1)
        for i in range(self.num_upsamples):
            x = leaky_relu(x, 0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)                              # nn.ReflectionPad1d((1, 0))
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x, s)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = leaky_relu(x, 0.01)                                                            # F.leaky_relu's default slope
        x = self.conv_post(x)
        half = self.post_n_fft // 2 + 1
        spec = mx.exp(x[:, :half, :])
        phase = mx.sin(x[:, half:, :])
        return self.stft.inverse(spec, phase)


class AdainResBlk1d:
    """istftnet.py 340-381. With upsample, the residual path doubles the length with a depthwise transposed
    conv (`pool`) and the shortcut with a nearest-neighbour repeat."""

    def __init__(self, w, name, dim_in, dim_out, upsample=False):
        self.w, self.name, self.upsample = w, name, upsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = Conv1d(w, f"{name}.conv1", padding=1)
        self.conv2 = Conv1d(w, f"{name}.conv2", padding=1)
        if self.learned_sc:
            self.conv1x1 = Conv1d(w, f"{name}.conv1x1")
        if upsample:
            self.pool = ConvTranspose1d(w, f"{name}.pool", stride=2, padding=1, output_padding=1, groups=dim_in)

    def _shortcut(self, x):
        if self.upsample:
            x = upsample_nearest(x, 2)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = adain(x, s, self.w, f"{self.name}.norm1")
        x = leaky_relu(x, 0.2)
        if self.upsample:
            x = self.pool(x)
        x = self.conv1(x)
        x = adain(x, s, self.w, f"{self.name}.norm2")
        x = leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x

    def __call__(self, x, s):
        return (self._residual(x, s) + self._shortcut(x)) * (1.0 / math.sqrt(2.0))


class Decoder:
    """istftnet.py 384-421."""

    def __init__(self, w, dim_in, prefix="decoder", **istftnet):
        self.w, self.p = w, prefix
        self.encode = AdainResBlk1d(w, f"{prefix}.encode", dim_in + 2, 1024)
        self.decode = [AdainResBlk1d(w, f"{prefix}.decode.{i}", 1024 + 2 + 64, 1024) for i in range(3)]
        self.decode.append(AdainResBlk1d(w, f"{prefix}.decode.3", 1024 + 2 + 64, 512, upsample=True))
        self.F0_conv = Conv1d(w, f"{prefix}.F0_conv", stride=2, padding=1)
        self.N_conv = Conv1d(w, f"{prefix}.N_conv", stride=2, padding=1)
        self.asr_res = Conv1d(w, f"{prefix}.asr_res.0")
        self.generator = Generator(w, f"{prefix}.generator", **istftnet)

    def features(self, asr, F0_curve, N, s):
        """Everything before the generator: [B, 512, 2 * frames]."""
        F0 = self.F0_conv(F0_curve[:, None, :])
        N = self.N_conv(N[:, None, :])
        x = mx.concatenate([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = mx.concatenate([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample:
                res = False
        return x

    def __call__(self, asr, F0_curve, N, s, draws=None):
        return self.generator(self.features(asr, F0_curve, N, s), s, F0_curve, draws)

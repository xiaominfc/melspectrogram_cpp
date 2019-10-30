#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 xiaominfc(武汉鸣鸾信息科技有限公司) <xiaominfc@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""


import numpy as np
from numpy.lib.stride_tricks import as_strided
MAX_MEM_BLOCK = 2**8 * 2**10

### hann window

def _len_guards(M):
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    if needed:
        return w[:-1]
    else:
        return w



def general_cosine(M, a, sym=True):
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)
    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return _truncate(w, needs_trunc)


def general_hamming(M, alpha, sym=True):
    return general_cosine(M, [alpha, 1. - alpha], sym)


def hann(M, sym=True):
    return general_cosine(M, [0.5, 0.5], sym)
    #return general_hamming(M, 0.5, sym)


#### end window



#### filters
def fft_frequencies(sr=22050, n_fft=2048):

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def hz_to_mel(frequencies, htk=False):
    frequencies = np.asanyarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels

def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def filters_mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):
    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights

#### end filters


def frame(x, frame_length=2048, hop_length=512, axis=-1):
    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)
    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize
    shape = list(x.shape)[:-1] + [frame_length, n_frames]
    strides = list(strides) + [hop_length * new_stride]
    result = as_strided(x, shape=shape, strides=strides)
    return result



def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)




# fft

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    fft_window = hann(win_length,sym=False)
    fft_window = pad_center(fft_window, n_fft)
    fft_window = fft_window.reshape((-1, 1))
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode) # 两边各补512个 对称

    
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length) 
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')
    fft = np.fft
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))
    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        data = y_frames[:, bl_s:bl_t] *fft_window
        stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window *
                                             y_frames[:, bl_s:bl_t],
                                             axis=0)
    return stft_matrix


def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,power=2.0, **kwargs):
    S = stft(y,n_fft=1024,hop_length=hop_length)
    S = np.abs(S)**power
    mel_basis = filters_mel(sr,n_fft=n_fft,**kwargs)
    return np.dot(mel_basis, S)

            


if __name__ == '__main__':
    import librosa
    #y, sr = librosa.load('./5016672465406.wav',sr=None)
    y, sr = librosa.load('./6983123609037.wav',sr=None)
    n_fft = 1024
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,fmax=8000,n_fft=n_fft)
    print(mels.T)
    print("=============")
    print(melspectrogram(y,sr=sr,n_fft=n_fft,n_mels=80,fmax=8000).T)


















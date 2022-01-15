import librosa
import numpy as np
from scipy.fftpack import dct
import math
import python_speech_features as sf

# If you want to see the spectrogram picture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_spectrogram(spec, note,file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """ 
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """

    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum

def get_mel_filter_weights(fft_bins):
    filter_bank_size = int(fft_len/2) + 1
    filter_weights   = np.zeros(filter_bank_size)
    for i in range(num_filter + 1):
        left   = int(fft_bins[i])
        center = int(fft_bins[i+1])
        k      = 1/(center - left)
        for j in range(left, center):
            filter_weights[j] = (j - left) * k
    # plt.plot(filter_weights)
    # plt.savefig('filter_weights.jpg')
    # plt.clf()
    return filter_weights

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """

    spectrum = np.square(spectrum)/fft_len
    low_fre = 0
    max_fre = fs/2
    low_mel = 2595*np.log10(1 + low_fre/700)
    max_mel = 2595*np.log10(1 + max_fre/700)

    mel_point = np.linspace(low_mel, max_mel, num = num_filter + 2) # 等间隔产生梅尔频率
    fre_point = 700 * (10**(mel_point/2595) - 1) # 将等间隔的梅尔频域换算成频域
    fft_bins  = fre_point * (fft_len + 1) / fs   # 频域换算成对应的fft bin
    filter_weights = get_mel_filter_weights(fft_bins)

    feats=np.zeros((spectrum.shape[0], num_filter))
    for i in range(spectrum.shape[0]):
        for j in range(num_filter+1):
            left   = int(fft_bins[j])
            center = int(fft_bins[j+1])
            for k in range(left, center):
                if(j < num_filter):
                    feats[i][j] += filter_weights[k] * spectrum[i][k]
                if(j > 0):
                    feats[i][j-1] += (1 - filter_weights[k]) * spectrum[i][k]
    # import pdb;pdb.set_trace()
    feats = np.log(feats)

    """
        FINISH by YOURSELF
    """
    return feats

def dct_ii(x):
    """参考：https://en.wikipedia.org/wiki/Discrete_cosine_transform
    """
    M = x.shape[0]
    y = np.zeros(M)
    for i in range(M):
        for j in range(M):
            y[i] += x[j] * np.cos(np.pi * i * (j + 0.5) / M)
    y = np.sqrt(2/M)*y
    return y


def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    feats = np.zeros((fbank.shape[0],num_mfcc))

    for i in range(fbank.shape[0]):
        feats[i] = dct_ii(fbank[i])[1:num_mfcc+1]

    """
        FINISH by YOURSELF
    """
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()

def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(fbank_feats, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()

# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
from lab1_tools import lifter
from lab1_tools import tidigit2labels
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fftpack import fft
from lab1_tools import *
from scipy.fftpack import dct
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

def mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    tmp = lifter(ceps, liftercoeff)
    # plt.pcolormesh(tmp.T)
    # plt.show()
    return tmp


# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    num_frames = 1 + int((len(samples) - winlen) / winshift)

    frames = np.zeros((num_frames, winlen))

    for i in range(num_frames):
        frames[i] = samples[i * winshift: i * winshift + winlen]

    # plt.pcolormesh(frames.T)
    # plt.show()

    return frames


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    b = [1, -p]
    a = 1

    output = np.apply_along_axis(lambda x: lfilter(b, a, x), 1, input)

    # plt.pcolormesh(output.T)
    # plt.show()

    return output


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    window = hamming(input.shape[1], sym=False)

    output = input * window

    # plt.pcolormesh(output.T)
    # plt.show()

    return output


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft_result = fft(input, n=nfft)

    power_spectrum = (np.abs(fft_result) ** 2)

    # power_spectrum = power_spectrum[:, :nfft // 2 + 1]

    # plt.pcolormesh(power_spectrum.T)
    # plt.show()
    return power_spectrum


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    bank = trfbank(samplingrate, input.shape[1])
    result = input @ bank.T
    # result = np.ones(input.shape)
    # for i, x in enumerate(input):
    #     for b in bank:
    #         result[i] = x*b
    # plt.show()

    output = np.log(result)

    # plt.figure(figsize=(100, 20))
    # plt.pcolormesh(output.T)
    # plt.show()
    return output


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    output = dct(input)
    output = output[:,:nceps]
    output = np.asarray(output)
    # plt.pcolormesh(output.T)
    # plt.show()
    return output


def get_all_data():
    data = np.load('lab1_data.npz', allow_pickle=True)['data']
    result_mfcc=[]
    result_mspec=[]
    #Concatenate all the MFCC frames from all utterances in the data array into a big feature [N × M] array where N is the total number of frames in the data set and M is the number of coefficients.
    for i, d in enumerate(data):
        sample=d['samples']
        samp_rate=d['samplingrate']
        result_mfcc.append(mfcc(sample,samplingrate=samp_rate))
        result_mspec.append(mspec(sample,samplingrate=samp_rate))
    return result_mfcc,result_mspec




def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    loc_dist = np.zeros((x.shape[0], y.shape[0]))
    acc_dist = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            loc_dist[i][j] = dist(x[i], y[j])
    for i in range(1, x.shape[0]):
        acc_dist[i][0] = acc_dist[i - 1][0] + loc_dist[i][0]
    for j in range(1, y.shape[0]):
        acc_dist[0][j] = acc_dist[0][j - 1] + loc_dist[0][j]
   
    for i in range(1, x.shape[0]):
        for j in range(1, y.shape[0]):
            acc_dist[i][j] = loc_dist[i][j] + min(acc_dist[i - 1][j], acc_dist[i][j - 1], acc_dist[i - 1][j - 1])
    
    path = []
    i = x.shape[0] - 1
    j = y.shape[0] - 1
    while i > 0 and j > 0:
        path.append((i, j))
        if acc_dist[i - 1][j] == min(acc_dist[i - 1][j], acc_dist[i][j - 1], acc_dist[i - 1][j - 1]):
            i -= 1
        elif acc_dist[i][j - 1] == min(acc_dist[i - 1][j], acc_dist[i][j - 1], acc_dist[i - 1][j - 1]):
            j -= 1
        else:
            i -= 1
            j -= 1
    path.append((i, j))
    path.reverse()
    return acc_dist[-1][-1] / (x.shape[0] + y.shape[0]), loc_dist, acc_dist, path
    



def GMM(mfcc_data,mfcc_uttarances):
    components=[4,8,16,32]
    data = np.load('lab1_data.npz', allow_pickle=True)['data']
    utt=[16,17,38,25]
    for c in components:
        gmm = GaussianMixture(n_components=c, covariance_type='diag')
        gmm.fit(mfcc_data)
        for u in utt:
            posterior = gmm.predict_proba(mfcc_uttarances[u])
            plot_posterior(posterior, data, u)




def plot_posterior(posterior, data, utt):
    sum_posterior = np.sum(posterior, axis=0).reshape(1,-1)
    sum_posterior /= np.sum(posterior)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data[utt]['samples'])
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.subplot(1, 2, 2)
    plt.imshow(posterior.T, aspect='auto', origin='lower')
    plt.xlabel('Frame index')
    plt.ylabel('Gaussian component')
    plt.title('Posterior probabilities')
    plt.tight_layout()
    plt.show()

def hierarchical_clustering(distance_matrix, labels):
    """Hierarchical clustering.

    Args:
        distance_matrix: NxN matrix of pairwise distances between N sequences
        labels: N labels for the sequences

    Outputs:
        Z: linkage matrix
        T: dendrogram tree
        fig: figure handle

    Note that you only need to define the first output for this exercise.
    """
    Z = linkage(distance_matrix, 'single')
    T = dendrogram(Z,labels=labels)
    fig = plt.gcf()
    plt.show()
    return Z, T, fig
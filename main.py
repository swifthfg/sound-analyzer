import matplotlib.pyplot as plt
from numpy import fft as fft
import numpy as np
import scipy.io.wavfile
import pydub
import os

# to get rid of the error saying the plotlib cannot draw that much data points
plt.rcParams['agg.path.chunksize'] = 10000

EXTENSION_WAV = '.wav'


def convertMP3toWAV(fileMP3Path):
    baseName = os.path.basename(fileMP3Path)
    wavName = os.path.splitext(baseName)[0] + EXTENSION_WAV
    mp3File = pydub.AudioSegment.from_mp3(fileMP3Path)
    mp3File.export(wavName, format="wav")


def drawFastFourierTransformedSoundGraph(audioData):
    fourier = fft.fft(audioData)
    plt.plot(fourier, color='#ff7f00')
    plt.xlabel('k')
    plt.ylabel('Amplitude')
    plt.show()


def drawAmplitudeOverTimeGraph(rate, audioData):
    time = np.arange(0, float(audioData.shape[0]), 1) / rate
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, audioData, linewidth=0.01, alpha=0.7, color='#ff7f00')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def getLeftChannelDataOfSound(audioData):
    # first column
    return audioData[:, 0]


def getRightChannelDataOfSound(audioData):
    # second column
    return audioData[:, 1]


def getTrackTimeInSeconds(rate, audioData):
    # length of the track = allDataPoints / dataPointsPerSecond
    return audioData.shape[0] / rate


def main():
    print('Start analyzing')
    fileWAV = 'Respect.wav'

    rate, audioData = scipy.io.wavfile.read(fileWAV)
    print('Rate: ' + str(rate))  # how many data points are there per second
    print('Audio shape: ' + str(audioData.shape))
    print('Length of music in seconds: ' + str(audioData.shape[0] / rate))
    print('Number of mono/stereo channels: ' + str(audioData.shape[1]))

    time = np.arange(0, float(audioData.shape[0]), 1) / rate

    # drawAmplitudeOverTimeGraph(rate, audioData)


if __name__ == '__main__':
    main()
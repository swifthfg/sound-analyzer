import matplotlib.pyplot as plt
from numpy import fft as fft
import scipy.io.wavfile
import numpy as np
import pydub
import os

# to get rid of the error saying the plotlib cannot draw that much data points
plt.rcParams['agg.path.chunksize'] = 10000

EXTENSION_WAV = '.wav'
PLOT_ALPHA = 0.8
NP_AXIS_ROW = 0
NP_AXIS_COL = 1


def convertMP3toWAV(fileMP3Path):
    baseName = os.path.basename(fileMP3Path)
    wavName = os.path.splitext(baseName)[0] + EXTENSION_WAV
    mp3File = pydub.AudioSegment.from_mp3(fileMP3Path)
    mp3File.export(wavName, format="wav")


def drawFastFourierTransformedSoundGraph(audioData):
    fourier = fft.fft(audioData)
    plt.plot(fourier, alpha=PLOT_ALPHA, color='#ff7f00')
    plt.xlabel('c')
    plt.ylabel('Amplitude')
    plt.show()


def drawAmplitudeOverTimeGraph(rate, audioData):
    time = np.arange(0, float(audioData.shape[0]), 1) / rate
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, audioData, linewidth=0.01, alpha=PLOT_ALPHA, color='#ff7f00')
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


def getPitchPointsTimeData(rate, audioData):
    time = np.arange(0, float(audioData.shape[0]), 1) / rate
    print(len(time))
    data = getDataPointsInSpecifiedTimeInterval(rate, audioData, 2, 3)
    for i in range(1000):
        print(data[i])


def getDataPointsInSpecifiedTimeInterval(rate, audioData, startSecond, endSecond):
    startIndex = rate * startSecond
    endIndex = rate * endSecond
    return audioData[startIndex:endIndex]


def getVarianceOfData(audioData):
    return np.var(audioData, axis=NP_AXIS_ROW)


def getAmplitudeMagnitudeInSecond(audioData):
    return np.sum(np.abs(audioData))


def getAmplitudeMagnitudeForAllSeconds(rate, audioData):
    totalTime = getTrackTimeInSeconds(rate, audioData)
    resList = []
    for i in range(int(totalTime)):
        dataPoints = getDataPointsInSpecifiedTimeInterval(rate, audioData, i, i+1)
        resList.append((i, getAmplitudeMagnitudeInSecond(dataPoints)))
    resNP = np.array(resList)

    # sort in-place according to f1 field corresponding to second column(amplitude sum)
    resNP.view('i8, i8')[::-1].sort(order=['f1'], axis=NP_AXIS_ROW)
    return resNP


def main():
    print('Start analyzing')
    fileWAV = 'Respect.wav'

    rate, audioData = scipy.io.wavfile.read(fileWAV)
    print('Rate: ' + str(rate))  # how many data points are there per second
    print('Audio shape: ' + str(audioData.shape))
    print('Length of music in seconds: ' + str(audioData.shape[0] / rate))
    print('Number of mono/stereo channels: ' + str(audioData.shape[1]))

    getPitchPointsTimeData(rate, audioData)
    # print(getAmplitudeMagnitudeForAllSeconds(rate, audioData))

    # print(getVarianceOfData(audioData))

    # drawAmplitudeOverTimeGraph(rate, audioData)


if __name__ == '__main__':
    main()
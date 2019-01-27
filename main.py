import matplotlib.pyplot as plt
from numpy import fft as fft
import scipy.io.wavfile
import numpy as np
import decimal
import pydub
import boto3
import time
import os

from dotenv import load_dotenv
load_dotenv()

# to get rid of the error saying the plotlib cannot draw that much data points
plt.rcParams['agg.path.chunksize'] = 10000

EXTENSION_WAV = '.wav'
PLOT_ALPHA = 0.8
NP_AXIS_ROW = 0
NP_AXIS_COL = 1
PITCH_CONST = 0.6


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


def getDataPointsInSpecifiedTimeInterval(rate, audioData, startSecond, endSecond):
    startIndex = rate * startSecond
    endIndex = rate * endSecond
    return audioData[int(startIndex):int(endIndex)]


def getVarianceOfData(audioData):
    return np.var(audioData, axis=NP_AXIS_ROW)


def getAmplitudeMagnitudeInSecond(audioData):
    return np.sum(np.abs(audioData))


def getMaxOfAmplitudes(audioData):
    leftChannel = getLeftChannelDataOfSound(audioData)
    return leftChannel.max()


def getAmplitudeMagnitudeForAllSeconds(rate, audioData):
    totalTime = getTrackTimeInSeconds(rate, audioData)
    resList = []
    i = 0
    while i < totalTime:
        dataPoints = getDataPointsInSpecifiedTimeInterval(rate, audioData, i, i+0.1)
        # resList.append((i, getAmplitudeMagnitudeInSecond(dataPoints)))
        resList.append((i, getMaxOfAmplitudes(dataPoints)))
        i += 0.1
    resNP = np.array(resList, dtype=float)

    # sort in-place according to f1 field corresponding to second column(amplitude sum)
    resNP.view('i8, f8').sort(order=['f0'], axis=NP_AXIS_ROW)
    return resNP


def normalizeAmplitudeTotals(timeAmplitudeData):
    onlyMagnitude = timeAmplitudeData[:, 1]
    normalized = onlyMagnitude / float(onlyMagnitude.max())
    timeAmplitudeData[:, 1] = normalized
    return timeAmplitudeData


def getPitchPoints(normalizedData):
    return normalizedData[normalizedData[:, 1] > PITCH_CONST]


def main():
    try:
        sqsClient = boto3.client('sqs')
        dynamodb = boto3.resource('dynamodb')
        peakTable = dynamodb.Table(os.getenv('dynamodb_table'))

        print('Start analyzing')
        fileWAV = 'Respect.wav'

        rate, audioData = scipy.io.wavfile.read(fileWAV)
        print('Rate: ' + str(rate))  # how many data points are there per second
        print('Audio shape: ' + str(audioData.shape))
        print('Length of music in seconds: ' + str(audioData.shape[0] / rate))
        print('Number of mono/stereo channels: ' + str(audioData.shape[1]))

        timeAmplitudeData = getAmplitudeMagnitudeForAllSeconds(rate, audioData)
        peakData = getPitchPoints(normalizeAmplitudeTotals(timeAmplitudeData))
        peakList = peakData.tolist()
        decimalList = list(map(lambda x: [decimal.Decimal(str(x[0])), decimal.Decimal(str(x[1]))], peakList))

        dynamoRes = peakTable.put_item(
            Item={
                'id': str(time.time()),
                'filename': fileWAV,
                'rate': rate,
                'peakData': decimalList
            }
        )
        print(dynamoRes)
        # print(peakData)
    except Exception as e:
        print('Error:')
        print(e)



if __name__ == '__main__':
    main()
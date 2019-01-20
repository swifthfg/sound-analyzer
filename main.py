import mutagen
import scipy.io.wavfile
import pydub
import os

EXTENSION_WAV = '.wav'

def convertMP3toWAV(fileMP3Path):
    fileName = os.path.basename(fileMP3Path)
    wavName = os.path.splitext(fileName)[0] + EXTENSION_WAV

    mp3File = pydub.AudioSegment.from_mp3(fileMP3Path)
    mp3File.export(wavName, format="wav")

def main():
    print('Start analyzing')
    fileWAV = 'Respect.wav'

    rate, audioData = scipy.io.wavfile.read(fileWAV)
    print('Rate: ' + str(rate))  # how many data points are there per second
    print('Audio shape: ' + str(audioData.shape))
    print('Lengt of music in seconds: ' + str(audioData.shape[0] / rate))  # length of the track = allDataPoints / dataPointsPerSecond





if __name__ == '__main__':
    main()
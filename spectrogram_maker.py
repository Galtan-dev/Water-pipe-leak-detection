# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile
import scipy.signal as signal

fs, audio = wavfile.read('video_and_sound.wav')

plt.specgram(audio, Fs=fs)

plt.show()
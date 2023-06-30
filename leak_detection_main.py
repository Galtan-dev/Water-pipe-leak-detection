from scipy.io import wavfile
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import csv
import pandas as pd

# open file with sound data
sampling_frequency, sound_data = wavfile.read('video_and_sound.wav')

# inputs one by one - continuously drifted by one
in_1 = []
in_2 = []
in_3 = []
in_4 = []
in_5 = []
target = []
for i in range(0, 8565114):
    in_1.append(sound_data[i])
for i in range(1, 8565115):
    in_2.append(sound_data[i])
for i in range(2, 8565116):
    in_3.append(sound_data[i])
for i in range(3, 8565117):
    in_4.append(sound_data[i])
for i in range(4, 8565118):
    in_5.append(sound_data[i])
for i in range(5, 8565119):
    target.append(sound_data[i])

# creation of matrix of inputs and target vector
in_matrix = []
target_matrix = []
for i in range(0, 8565114):
    in_matrix.append(in_1[i])
for i in range(0, 8565114):
    in_matrix.append(in_2[i])
for i in range(0, 8565114):
    in_matrix.append(in_3[i])
for i in range(0, 8565114):
    in_matrix.append(in_4[i])
for i in range(0, 8565114):
    in_matrix.append(in_5[i])
for i in range(0, 8565114):
    target_matrix.append(target[i])

in_matrix = np.asarray(in_matrix)
in_matrix = np.reshape(in_matrix, [8565114, 5], order="F")

d = np.reshape(target_matrix, (8565114,))
x = np.reshape(in_matrix, (8565114, 5))

# detection
# f = pa.filters.FilterGNGD(n=1, mu=0.40, w="zeros")
f = pa.filters.FilterAP(n=5, order=5, mu=0.50, ifc=0.005, w="zeros")
y, e, w = f.run(d, x)

le = pa.detection.learning_entropy(w, m=50, order=1)
# le = pa.detection.ELBND(E, w, function="max")

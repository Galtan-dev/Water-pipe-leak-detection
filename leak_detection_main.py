import numpy as np
import matplotlib.pylab as plt
import padasip as pa

# open data
with open("in_sound_data.csv", "r", encoding="utf-8") as in_sound_data:
    in_matrix = (np.genfromtxt(in_sound_data, delimiter=",", skip_header=0))

with open("target_sound_data.csv", "r", encoding="utf-8") as target_sound_data:
    target_matrix = (np.genfromtxt(target_sound_data, delimiter=",", skip_header=0))
k = len(target_matrix)
d = np.reshape(target_matrix, (k,))
x = np.reshape(in_matrix, (k, 5))

# detection
# f = pa.filters.FilterGNGD(n=5, mu=0.50, w="zeros")
f = pa.filters.FilterAP(n=5, order=5, mu=0.50, ifc=0.005, w="zeros")
y, e, w = f.run(d, x)

# det = pa.detection.learning_entropy(w, m=50, order=1, alpha=[8., 9., 10., 11., 12., 13.])
# det = pa.detection.ELBND(w, e, function="max")
det = pa.detection.ESE(w)

# ploting
fig, axs = plt.subplots(2, 1)
axs[0].plot(d)
axs[1].plot(det)
fig.tight_layout()
plt.show()















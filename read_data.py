import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("..\\data\\M3\\M3C.csv", delimiter=",")
print data.shape, data

plt.ion()
plt.show()

i = 0
for serie in reversed(data):
    i += 1
    print i

    y = serie[np.logical_not(np.isnan(serie))]
    plt.clf()
    plt.plot(y)
    plt.draw()
    plt.pause(0.01)

import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load('gaus_wavelet_2.npy')
    plt.plot(data[0])
    plt.xlim(0,100)
    plt.show()

if __name__ == '__main__':
    main()


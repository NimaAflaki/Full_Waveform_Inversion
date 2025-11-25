import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    
    fc = 1e6 # Carrier frequency
    RF = np.load('3212RF.npy') # Load RF data
    t_step = 0.0457e-6 # time step

    t = np.linspace(0, (len(RF) - 1) * t_step, len(RF))

    # Multiply by carrier freq
    I = RF * np.cos(2 * np.pi * fc * t)
    Q = RF * np.sin(2 * np.pi * fc * t)

    # Apply LPF
    f_cutoff = 5e5 # Bandwidth/2
    nyquist = 1 / (2 * t_step)
    norm_cutoff = f_cutoff / nyquist
    b, a = signal.butter(4, norm_cutoff, btype='lowpass')
    I_filtered = signal.filtfilt(b, a, I)
    Q_filtered = signal.filtfilt(b, a, Q)
    IQ = I_filtered + Q_filtered * 1j
    
    # graph
    plt.plot(I_filtered[0::10])
    plt.show()
    plt.plot(Q_filtered[0::10])
    plt.show()
    plt.plot(IQ[0::10])
    plt.show()

    ## With Hilbert Transform
    IQh = signal.hilbert(RF)
    plt.plot(IQh.real[0::10])
    plt.show()
    plt.plot(IQh.imag[0::10])
    plt.show()
    plt.plot(np.abs(IQh)[0::10])
    plt.show()





if __name__ == "__main__":
    main()
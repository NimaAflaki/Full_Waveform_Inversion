import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gausspulse

# Parameters
f0 = 1e6       # center frequency (Hz)
fs = 100e6     # sampling frequency (Hz)
fracBW = 1.00  # fractional bandwidth of the transducer

# Cutoff time at -40 dB, with fractional BW defined at -6 dB
tc = gausspulse('cutoff', fc=f0, bw=fracBW, bwr=-6, tpr=-40)

# Time vector: start at 0 instead of centered around 0
t = np.arange(0, 2 * tc, 1/fs)

# Impulse response (Gaussian-modulated sinusoid)
impResp = gausspulse(t - tc, fc=f0, bw=fracBW, bwr=-6)

# --- Pad with zeros until total length = 4200 ---
target_len = 4200
impResp_padded = np.zeros(target_len)
impResp_padded[:len(impResp)] = impResp[:min(len(impResp), target_len)]

# Adjust the time vector accordingly
t_padded = np.arange(0, target_len) / fs

# Plot
plt.figure()
plt.plot(t_padded, impResp_padded)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Gaussian Pulse (Zero-padded to length 4200)')
plt.show()

print(f"Original length: {len(impResp)}, Padded length: {len(impResp_padded)}")
np.save('source_wavelet.npy', impResp_padded)

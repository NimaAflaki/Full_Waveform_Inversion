import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gausspulse

# --- Parameters ---
f0 = 1e6      # center frequency (Hz)
fs = 100e6    # sampling frequency (Hz)
fracBW = 1.00 # fractional bandwidth (–6 dB point)
cutoff_dB = -40
N = 12440

# --- Compute cutoff time at -40 dB ---
# Equivalent to MATLAB: tc = gauspuls('cutoff', f0, fracBW, -6, -40)
tc = gausspulse('cutoff', fc=f0, bw=fracBW, bwr=-6, tpr=cutoff_dB)

# --- Create time vector centered around zero ---
t = np.arange(-tc, tc, 1/fs)

# --- Generate Gaussian pulse ---
impResp = gausspulse(t, fc=f0, bw=fracBW, bwr=-6)
pad_len = N - len(impResp)
if pad_len > 0:
    impResp = np.pad(impResp, (0, pad_len))
else:
    impResp = impResp[:N]

np.save('wavelet.npy', impResp)

# --- Time vector for padded signal ---
t_full = np.arange(len(impResp)) / fs
# --- Plot ---
plt.figure(figsize=(10, 3))
plt.plot(t_full, impResp)
plt.title("Gaussian Pulse (1 MHz center, 100% BW, -40 dB cutoff)")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

print(f"Cutoff time (tc): {tc*1e6:.3f} µs")
print(f"Impulse response shape: {impResp.shape}")


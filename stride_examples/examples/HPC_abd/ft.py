from stride import ScalarField, Space, Time, Problem
import numpy as np
import matplotlib.pyplot as plt
import sys

## Set up problem space and time
shape = (356, 385)
extra = (50, 50)
absorbing = (40, 40)
spacing = (0.5e-3, 0.5e-3)
space = Space(shape=shape, extra=extra, absorbing=absorbing, spacing=spacing)

start = 0.
step = 0.08e-6
num = 2500
time = Time(start=start, step=step, num=num)
problem = Problem(name="fourier_analysis", space=space, time=time)

## Load velocity models
vp_true = ScalarField(name="true_model", grid=problem.grid)
vp_true.load('data/anastasio2D-TrueModel.h5')

vp_reconstruct = ScalarField(name="reconstructed_vp", grid=problem.grid)
vp_reconstruct.load(sys.argv[1])

## Compute raw error
vp_error = ScalarField(name="vp_error", grid=problem.grid)
vp_error._set_data(np.zeros_like(vp_true.extended_data))
vp_error.data[...] = vp_true.data - vp_reconstruct.data
raw_error = vp_error.data

## Apply filter (same as your smoothing logic)
error = np.copy(raw_error)
for i in range(1, shape[0] - 1):
    for j in range(1, shape[1] - 1):
        error[i, j] = (
            0.0625 * (raw_error[i-1, j-1] + raw_error[i-1, j+1] + raw_error[i+1, j-1] + raw_error[i+1, j+1]) +
            0.125  * (raw_error[i-1, j] + raw_error[i, j-1] + raw_error[i, j+1] + raw_error[i+1, j]) +
            0.25   *  raw_error[i, j]
        )

## Compute 2D Fourier Transform
fft_result = np.fft.fft2(error)
fft_shifted = np.fft.fftshift(fft_result)
magnitude_spectrum = np.abs(fft_shifted)

## Plot the magnitude spectrum
plt.figure(figsize=(8, 6))
plt.imshow(np.log1p(magnitude_spectrum), cmap='inferno', interpolation='nearest')
plt.colorbar(label='Log Magnitude')
plt.title("2D Fourier Transform: Magnitude Spectrum")
plt.xlabel("Frequency X")
plt.ylabel("Frequency Y")
plt.tight_layout()
plt.show()

## Print stats
print("Max FFT Magnitude:", np.max(magnitude_spectrum))
print("Mean FFT Magnitude:", np.mean(magnitude_spectrum))

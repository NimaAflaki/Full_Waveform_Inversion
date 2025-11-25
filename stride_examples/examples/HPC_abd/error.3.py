from stride import ScalarField, Space, Time, Problem
import numpy as np
import matplotlib.pyplot as plt
import sys


## Set up identical problem 
shape = (356, 385)
extra = (50, 50)
absorbing = (40, 40)
spacing = (0.5e-3, 0.5e-3)
space = Space(shape=shape,
                extra=extra,
                absorbing=absorbing,
                spacing=spacing)

start = 0.
step = 0.08e-6
num = 2500
time = Time(start=start,
            step=step,
            num=num)

problem = Problem(name="find_error", space=space, time=time)

## Find raw error
vp_true = ScalarField(name="true_model", grid=problem.grid)
vp_true.load('data/anastasio2D-TrueModel.h5')

vp_reconstruct = ScalarField(name="reconstructed_vp", grid=problem.grid)
vp_reconstruct.load(sys.argv[1])

vp_error = ScalarField(name="vp_error", grid=problem.grid)
vp_error._set_data(np.zeros_like(vp_true.extended_data))
vp_error.data[...] = vp_true.data - vp_reconstruct.data

###vp_reconstruct.plot()
###vp_true.plot()
###vp_error.plot()

## Clean raw error
raw_error = vp_error.data

plt.imshow(np.abs(raw_error), cmap='gray', interpolation='nearest')
plt.colorbar()
plt.title("Error plot")
#####plt.show()


avg_raw_error = np.average(raw_error)
print("Average of raw data: ", avg_raw_error)

avg_abs_raw_data = np.average(np.abs(raw_error))
print("Average of absolute value of raw data: ", avg_abs_raw_data)

rms_raw_error = np.sqrt(np.mean(raw_error**2))
print("RMS of raw data: ", rms_raw_error)


## Apply filter
error = np.copy(raw_error)
dim = raw_error.shape
print(dim)

for i, row in enumerate(error):
    for j, val in enumerate(row):
        if i == 0 or j == 0 or i == 355 or j == 384:
            continue
        error[i, j] = (
            0.0625 * (raw_error[i-1, j-1] + raw_error[i-1, j+1] + raw_error[i+1, j-1] + raw_error[i+1, j+1]) +
            0.125 *  (raw_error[i-1, j] + raw_error[i, j-1] + raw_error[i, j+1] + raw_error[i+1, j]) +
            0.25 *   (raw_error[i, j])
        )

avg = np.average(error)
print("Average filtered error: ", avg)

avg_abs_data = np.average(np.abs(error))
print("Average of absolute value of raw data: ", avg_abs_data)

rms = np.sqrt(np.mean(error**2))
print("RMS of filtered data: ", rms)


plt.imshow(np.abs(error), cmap='gray_r', interpolation='nearest')
plt.colorbar()
plt.title("Error plot")
plt.show()


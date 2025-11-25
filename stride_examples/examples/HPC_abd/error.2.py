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


args = sys.argv[1:]
for arg in args:
    ## Find raw error
    vp_true = np.load('truth_clean.npy')

    vp_model = arg
    file = vp_model + '_clean.npy'
    vp_constructed = np.load(file)
    error = vp_true - vp_constructed

    abs_error = np.abs(error)
    relative_error = np.divide(abs_error, vp_true, out=np.zeros_like(abs_error), where=vp_true!=0)

    avg_abs_raw_data = np.nanmean(np.abs(error))
    print("Average of absolute value of raw data: ", avg_abs_raw_data)

    print(np.nanmax(relative_error))

    plt.imshow(relative_error, cmap='gray_r', interpolation='bilinear', vmax=0.15, vmin=0)
    plt.colorbar()
    plt.title(vp_model + " Normalized Point-Wise Error")
    plt.show()



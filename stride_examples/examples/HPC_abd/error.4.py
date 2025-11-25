from stride import ScalarField, Space, Time, Problem
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.ndimage


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
    if arg == 'homo':
        ## Find raw error
        vp_true = np.load('ground_truth_mask.npy')

        error = vp_true[...] - 1500

        abs_error = np.abs(error)
        #relative_error = np.divide(abs_error, vp_true, out=np.zeros_like(abs_error), where=vp_true!=0)

        avg_abs_raw_data = np.nanmean(abs_error)

        n = 10
        kernel = np.ones((n, n)) / (n * n)
        
        abs_error_smoothed = scipy.ndimage.convolve(abs_error, kernel, mode='reflect')
        
        plt.imshow(abs_error_smoothed, cmap='cool', interpolation='nearest', vmax=40, vmin=0)
        plt.colorbar()
        plt.title("Homogeneous Starting Model Error")
        plt.show()
        continue
        
    ## Find raw error
    vp_true = np.load('ground_truth_mask.npy')

    vp_model = arg
    file = vp_model + '_mask.npy'
    vp_constructed = np.load(file)
    error = vp_true - vp_constructed

    abs_error = np.abs(error)
    relative_error = np.divide(abs_error, (vp_true[...]-1500), out=np.zeros_like(abs_error), where=vp_true!=0)
    abs_rel_error = np.abs(relative_error)
    
    plt.imshow(abs_rel_error, cmap='cool', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(vp_model + " Normalized Point-Wise Error")
    plt.show()



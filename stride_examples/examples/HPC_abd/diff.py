import sys
import numpy as np
import matplotlib.pyplot as plt
from stride import Space, Time, Problem, ScalarField
from stride.utils import fetch



def main():
    shape = (1637, 1228)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.2566666666666666666667e-3, 0.2566666666666666666667e-3)

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)
    start=0.0e-6
    step=0.08e-6
    num=4200

    time = Time(start=start,
                step=step,
                num=num)



    problem = Problem(name="find_error", space=space, time=time)

    file = sys.argv[1]
    file_name = file + '.h5'
    vp = ScalarField(name='vp', grid=problem.grid)
    fetch(file, dest=file_name)
    vp.load(file_name)
    file_2 = sys.argv[2]
    file_name_2 = file_2 + '.h5'
    vp_2 = ScalarField(name='vp_2', grid=problem.grid)
    fetch(file_2, dest=file_name_2)
    vp.load(file_name_2)
    print('start max = ', np.nanmax(vp.data), 'start_min = ', np.nanmin(vp.data))
    print('final max = ', np.nanmax(vp_2.data), 'final_min = ', np.nanmin(vp_2.data), 'final avg = ', np.mean(vp_2.data))
    diff = vp.data - vp_2.data
    print('Max = ', np.nanmax(diff), '\nMin = ', np.nanmin(diff))
    plt.imshow(np.abs(diff), cmap='gray_r', interpolation='nearest', vmax=500, vmin=0)
    plt.colorbar()
    plt.title(file)
    plt.show()




if __name__ == '__main__':
    main()

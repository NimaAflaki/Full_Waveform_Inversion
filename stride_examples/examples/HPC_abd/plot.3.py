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


    if sys.argv[1] == 'og':
        vp = ScalarField(name='vp', grid=problem.grid)
        fetch('anastasio2D', dest='data/anastasio2D-TrueModel.h5')
        vp.load('data/anastasio2D-TrueModel.h5')
        file = 'True Model'
    elif sys.argv[1] == 'index':
        indices = np.load('outside_index.npy')
        vp = ScalarField(name='vp', grid=problem.grid)
        fetch('anastasio2D', dest='data/anastasio2D-TrueModel.h5')
        vp.load('data/anastasio2D-TrueModel.h5')
        data = np.array(vp.data)
        print(data)
        index_set = set(map(tuple, indices))
        # Set outer values to NaN
        for i, j in index_set:
            data[i, j] = np.nan
        file = 'clean_truth'
        vp = data
        print(vp)
        print('Max: ', vp.max(), ' Min: ', vp.min())
        np.save('truth_clean.npy', vp)

    else:
        file = sys.argv[1]
        file_name = file + '.h5'
        vp = ScalarField(name='vp', grid=problem.grid)
        fetch(file, dest=file_name)
        vp.load(file_name)
    homo_data = np.array(vp.data)
    print('Homo:')
    print("    shape = ", homo_data.shape)
    print("    size  = ", homo_data.size)
    print('Max = ', np.max(vp.data), 'Min = ', np.min(vp.data), 'Avg = ', np.mean(vp.data))
    plt.imshow(vp.data, cmap='gray', interpolation='nearest', vmax=1900)
    plt.colorbar()
    plt.title(file)
    plt.show()




if __name__ == '__main__':
    main()

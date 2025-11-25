import numpy as np
import sys
import matplotlib.pyplot as plt
from stride import Space, Time, Problem, ScalarField
from stride.utils import fetch


def main():
        
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
    problem = Problem(name="cleaning", space=space, time=time)
    indices = np.load('high_index.npy')
    indices_prime = np.load('prime_index.npy')

    args = sys.argv[1:]
    for arg in args:
        if arg == 'og':
            vp = ScalarField(name='vp', grid=problem.grid)
            fetch('anastasio2D', dest='data/anastasio2D-TrueModel.h5')
            vp.load('data/anastasio2D-TrueModel.h5')
            vp_model = 'ground_truth' 
            data = np.array(vp.data)
            index_set = set(map(tuple, indices))
            for i, j in index_set:
                data[i, j] = np.nan
            index_set_2 = set(map(tuple, indices_prime))
            for i, j in index_set_2:
                data[i, j] = np.nan

            np.save('masked_vp/' + vp_model + '_mask.npy', data)

            plt.imshow(data, cmap='grey', interpolation='nearest', vmax=1560)
            plt.colorbar()
            plt.title(vp_model) 
            plt.show()
            continue
            
        vp_model = arg

        file = vp_model + '_vp.h5'
        vp = ScalarField(name='vp', grid=problem.grid)
        fetch(vp_model, dest=file)
        vp.load(file)

        data = np.array(vp.data)
        index_set = set(map(tuple, indices))
        for i, j in index_set:
            data[i, j] = np.nan
        index_set_2 = set(map(tuple, indices_prime))
        for i, j in index_set_2:
            data[i, j] = np.nan

        np.save('masked_vp/' + vp_model + '_mask.npy', data)

        plt.imshow(data, cmap='grey', interpolation='nearest', vmax=1560)
        plt.colorbar()
        plt.title(vp_model) 
        plt.show()




if __name__ == '__main__':
        main()
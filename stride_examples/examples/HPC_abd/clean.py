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
    indices = np.load('outside_index.npy')

    vp_model = sys.argv[1]
    file = vp_model + '_vp.h5'
    vp = ScalarField(name='vp', grid=problem.grid)
    fetch(vp_model, dest=file)
    vp.load(file)

    data = np.array(vp.data)
    index_set = set(map(tuple, indices))
    for i, j in index_set:
        data[i, j] = np.nan
    vp = data
    np.save('clean_vp/' + vp_model + '_clean.npy', vp)

    #plt.imshow(vp, cmap='gray', interpolation='nearest')
    #plt.colorbar()
    #plt.title(vp_model) 
    #plt.show()




if __name__ == '__main__':
        main()
from stride import ScalarField, Problem, Space, Time
from stride.utils import fetch
import numpy as np


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

    problem = Problem(name="find_error", space=space, time=time)
    vp = ScalarField(name='vp', grid=problem.grid)
    fetch('anastasio2D', dest='data/anastasio2D-TrueModel.h5')
    vp.load('data/anastasio2D-TrueModel.h5')
    data = np.array(vp.data)

    indices = np.load('outside_index.npy')
    indices = list(map(tuple, indices))
    more = np.argwhere(data > 1560)

    for idx in map(tuple, more):
        if idx not in indices:
            indices.append(idx)

    indices = np.array(indices)
    np.save('high_index.npy', indices)

if __name__ == '__main__':
    main()
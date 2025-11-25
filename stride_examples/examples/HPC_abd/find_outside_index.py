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


    indices = []
    i, j = 0, 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i == 0 or j == 0:
                indices.append((i,j))
            else:
                if (i-1, j) not in indices and (i, j-1) not in indices:
                    continue
                else:
                    if data[i, j] == 1500:
                        indices.append((i, j))

    # Reverse
    for i in range(data.shape[0]-1, -1, -1):
        for j in range(data.shape[1]-1, -1, -1):
            if (i, j) in indices:
                continue
            else:
                if (i+1, j) not in indices and (i, j+1) not in indices:
                    continue
                else:
                    if data[i, j] == 1500:
                        indices.append((i, j))



    indices_array = np.array(indices)
    np.save('outside_index', indices_array)
    print(np.shape(indices))
    print(np.shape(data))


if __name__ == '__main__':
    main()
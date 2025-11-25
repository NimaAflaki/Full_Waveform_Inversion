import numpy as np
import sys
from stride import *

async def main(runtime):
    # Create the grid
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

    grid = Grid(space=space, time=time)
    name = sys.argv[1]
    problem = Problem(name=name, space=space, time=time)

    #Get ScalarField
    vp = ScalarField(name='vp', grid=grid)
    vp.load(name + '.h5')
    data = np.array(vp.data)

    np.save(name + '.npy', data)

if __name__ == '__main__':
    mosaic.run(main)

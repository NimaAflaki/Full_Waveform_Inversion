import numpy as np
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
    problem = Problem(name='size', space=space, time=time)
    #Get homo size
    vp_homo = ScalarField.parameter(name='vp', grid=problem.grid)

    vp_homo.fill(1540.)
    homo_data = np.array(vp_homo.data)
    np.save('homo.npy', homo_data)

if __name__ == '__main__':
    mosaic.run(main)

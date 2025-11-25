from stride import *
from stride.utils import fetch, wavelets
import sys
import numpy as np


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

    # Create problem
    name = sys.argv[1]
    num_of_trans, degrees = name.split('_')
    num_of_trans = int(num_of_trans)
    degrees = int(degrees)
    arch = 3.14159 * 400e-3 / num_of_trans
    problem = Problem(name=name,
                      space=space, time=time)

    # Create medium
    gt_rect = np.load('data/gt_rect.npy')
    gt_rect = gt_rect.T
    gt_rect = np.where(gt_rect == 0, 1540, gt_rect)
    vp_gt = ScalarField(name='vp_gt', grid=grid, data=gt_rect)
    problem.medium.add(vp_gt)

    # Create transducers
    if degrees == 180:
        problem_fake = Problem(name='fake', space=space, time=time)
        problem_fake.transducers.default()
        radius = (200e-3, 200e-3)
        problem_fake.geometry.default('elliptical', num_of_trans, radius=radius)
        locations = problem_fake.geometry.locations
        sorted_by_y = sorted(locations, reverse=True, key=lambda loc: loc.coordinates[1])
        num = num_of_trans * degrees / 360
        num = int(num)
        semicircle_locations = sorted_by_y[:num]
        semicircle_coords = [loc.coordinates.tolist() for loc in semicircle_locations]
        shifted_coords = [[x, y - 50e-3] for x, y in semicircle_coords]
    elif degrees == 360:
        problem_fake = Problem(name='fake', space=space, time=time)
        problem_fake.transducers.default()
        radius = (200e-3, 200e-3)
        problem_fake.geometry.default('elliptical', num_of_trans, radius=radius)
        locations = problem_fake.geometry.locations
        sorted_by_y = sorted(locations, reverse=True, key=lambda loc: loc.coordinates[1])
        num = num_of_trans * 0.625
        num = int(num)
        semicircle_locations = sorted_by_y[:num]
        semicircle_coords = [loc.coordinates.tolist() for loc in semicircle_locations]
        shifted_coords = [[x, y - 50e-3] for x, y in semicircle_coords]

        if num_of_trans == 128 or num_of_trans == 256:
            start, final = shifted_coords[-1],  shifted_coords[-3]
            y = start[-1]
            x = start[0]
            while x < final[0]:
                shifted_coords.append([x, y])
                x = x + arch
        else:
            start, final = shifted_coords[-1],  shifted_coords[-2]
            y = start[-1]
            x = start[0]
            while x < final[0]:
                shifted_coords.append([x, y])
                x = x + arch

    i=0
    grid = Grid(space, time)
    for coord in shifted_coords:
        transducer = PointTransducer(id=i, grid=grid)
        problem.geometry.add(id=i, transducer=transducer, coordinates=coord)
        i += 1
    print("transducers added")


    # Create acquisitions
    problem.acquisitions.default()

    # Create wavelets
    f_centre = 0.50e6
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Run
    await forward(problem, pde, vp_gt)


if __name__ == '__main__':
    mosaic.run(main)


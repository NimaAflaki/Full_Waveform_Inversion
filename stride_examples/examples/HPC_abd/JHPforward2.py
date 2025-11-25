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
    step=0.0305e-6
    num=18660
    time = Time(start=start,
                step=step,
                num=num)

    acq_step = 0.457e-6
    num_acq = 1244
    frame_step = 1e-3
    num_frame = 1
    slow_time = SlowTime(acq_step=acq_step, num_acq=num_acq, frame_step=frame_step, num_frame=num_frame)

    grid = Grid(space=space, time=time, slow_time=slow_time)

    # Create problem
    name = sys.argv[1]
    problem = Problem(name=name,
                      space=space, time=time, slow_time=slow_time)

    # Create medium
    gt_rect = np.load('data/gt_rect.npy')
    gt_rect = gt_rect.T
    gt_rect = np.where(gt_rect == 0, 1540, gt_rect)
    vp_gt = ScalarField(name='vp_gt', grid=grid, data=gt_rect)
    problem.medium.add(vp_gt)

    # Create transducers
    locations = np.load('new_trans_locations.npy')
    i = 0
    for loc in locations:
        transducer = PointTransducer(id=i, grid=grid)
        problem.geometry.add(id=i, transducer=transducer, coordinates=loc)
        i += 1
    problem.plot()



    # Create acquisitions
    num_transducers = len(locations)
    for shot_id in range(num_transducers):
        shot = Shot(id=shot_id, problem=problem)

        # Add source
        shot.sources.append(shot_id)

        # Add all receivers
        for receiver_id in range(num_transducers):
            shot.receivers.append(receiver_id)

        # Create observed Traces with slow_time grid
        shot.observed = Traces(name=f'observed_{shot_id}', 
                              grid=Grid(space=space, time=slow_time),
                              num=num_transducers)
    # Create wavelets
    f_centre = 1e6
    n_cycles = 1

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step, envelope='gaussian')
    shot.plot()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Run
    await forward(problem, pde, vp_gt)


if __name__ == '__main__':
    mosaic.run(main)


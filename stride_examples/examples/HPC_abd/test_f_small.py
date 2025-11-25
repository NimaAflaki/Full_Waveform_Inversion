
from stride import *
from stride.utils import fetch, wavelets
import sys


async def main(runtime):
    # Create the grid
    space = Space(shape=(1637, 1228), extra=(50, 50), absorbing=(40, 40), spacing=(0.2566666666666666666667e-3, 0.2566666666666666666667e-3))
    time = Time(start=0.0e-6, step=0.08e-6, num=4200)

    grid = Grid(space, time)

    # Create problem
    name = sys.argv[1]

    problem = Problem(name=name,
                      space=space, time=time)

    # Create medium
    gt_rect = np.load('/home/nima/JHPstuff/ScanConversionSoSMap/abd_sample0014/gt_rect.npy')
    gt_rect = gt_rect.T

    gt_rect = np.where(gt_rect == 0, 1540, gt_rect)

    vp_gt = ScalarField(name='vp_gt', grid=grid, data=gt_rect)

    problem.medium.add(vp_gt)


    problem.medium.add(vp_gt)

    # Create transducers
    x_mm_transducer = np.load('/home/nima/JHPstuff/ScanConversionSoSMap/abd_sample0014/x_mm_transducer.npy')
    z_mm_transducer = np.load('/home/nima/JHPstuff/ScanConversionSoSMap/abd_sample0014/z_mm_transducer.npy')

    x_mm_transducer = (x_mm_transducer + 210)*1e-3
    z_mm_transducer = (z_mm_transducer + 105)*1e-3

    trans_coords = np.array(list(zip(x_mm_transducer, z_mm_transducer)))
    i = 0

    for coordinate in trans_coords:
        problem.geometry.add(id=i, transducer=PointTransducer(id=i, grid=grid), coordinates=coordinate) 

    # Create acquisitions
    problem.acquisitions.default()

    # Create wavelets
    f_centre = 0.50e6
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles, time.num, time.step)

    # Plot
    problem.plot()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Run
    await forward(problem, pde, vp_gt)

    





if __name__ == '__main__':
    mosaic.run(main)

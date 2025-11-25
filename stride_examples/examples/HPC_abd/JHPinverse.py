
from stride import *
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
    step=0.0457e-6
    num=12440

    time = Time(start=start,
                step=step,
                num=num)

    grid = Grid(space=space, time=time)

    # Create problem
    name = sys.argv[1]
    x = sys.argv[2]
    problem = Problem(name=name,
                      space=space, time=time)

    # Create medium
    if x == 'homo':
    	vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
        vp.fill(1540.)
    	problem.medium.add(vp)
    elif x == 'NN_mask':
        pred_rect = np.load('data/pred_rect.npy')
        pred_rect = pred_rect.T
        pred_rect = np.where(pred_rect == 0, 1540, pred_rect)
        pred_rect = np.where(pred_rect < 2000, 1540, pred_rect)
        vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
        vp.data[...] = pred_rect
        vp.pad()
        problem.medium.add(vp)
        problem.plot()
        vp.extended_plot()
    elif x == 'NN':
        pred_rect = np.load('data/pred_rect.npy')
        pred_rect = pred_rect.T
        pred_rect = np.where(pred_rect == 0, 1540, pred_rect)
        vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
        vp.data[...] = pred_rect
        vp.pad()
        problem.medium.add(vp)
        problem.plot()
        vp.extended_plot()


    # Create transducers
    locations = np.load('new_trans_locations.npy')
    i = 0
    grid = Grid(space, time)
    for loc in locations:
        transducer = PointTransducer(id=i, grid=grid)
        problem.geometry.add(id=i, transducer=transducer, coordinates=loc)
        i += 1
    problem.plot()
    # Create acquisitions
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Upsampling data
    new_step = 0.0457e-6
    new_num = 12440
    problem.time_resample(new_step=new_step, new_num=new_num)

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Create loss
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Create optimiser
    step_size = 10
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400., max=3000.)

    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model)

    # Run optimisation
    optimisation_loop = OptimisationLoop()


    max_freqs = [0.2e6, 0.3e6, 0.4e6, 0.5e6]

    num_blocks = len(max_freqs)
    num_iters = 8

    for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
        await adjoint(problem, pde, loss,
                      optimisation_loop, optimiser, vp,
                      num_iters=num_iters,
                      select_shots=dict(num=16, randomly=True),
                      f_max=freq, max_freqs=max_freqs, mute_traces=False)


if __name__ == '__main__':
    mosaic.run(main)


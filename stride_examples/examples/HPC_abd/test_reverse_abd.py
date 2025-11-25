from stride import *
import sys


async def main(runtime):
    space = Space(shape=(1637, 1228), extra=(50, 50), absorbing=(40, 40), spacing=(0.2566666666666666666667e-3, 0.2566666666666666666667e-3))
    time = Time(start=0.0e-6, step=0.08e-6, num=4200)

    grid = Grid(space, time)

    # Create problem
    name = sys.argv[1]

    problem = Problem(name=name,
                      space=space, time=time)


    # Create medium
    vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
    vp.fill(1540.)

    problem.medium.add(vp)

    # Create transducers
    x_mm_transducer = np.load('/home/nima/JHPstuff/ScanConversionSoSMap/abd_sample0014/x_mm_transducer.npy')
    z_mm_transducer = np.load('/home/nima/JHPstuff/ScanConversionSoSMap/abd_sample0014/z_mm_transducer.npy')

    x_mm_transducer = (x_mm_transducer + 210)*1e-3
    z_mm_transducer = (z_mm_transducer + 105)*1e-3

    trans_coords = np.array(list(zip(x_mm_transducer, z_mm_transducer)))
    i = 0

    for coordinate in trans_coords:
        problem.geometry.add(id=i, transducer=PointTransducer(id=i, grid=grid), coordinates=coordinate) 
        i = i + 1
        
    # Create acquisitions
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Plot
    problem.plot()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Create loss
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Create optimiser
    step_size = 10
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400., max=1700.)

    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model)

    # Run optimisation
    optimisation_loop = OptimisationLoop()

    max_freqs = [0.3e6, 0.4e6, 0.5e6]

    num_blocks = len(max_freqs)
    num_iters = 8

    for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
        await adjoint(problem, pde, loss,
                      optimisation_loop, optimiser, vp,
                      num_iters=num_iters,
                      select_shots=dict(num=16, randomly=True),
                      f_max=freq, max_freqs=max_freqs)

    vp.plot()

if __name__ == '__main__':
    mosaic.run(main)

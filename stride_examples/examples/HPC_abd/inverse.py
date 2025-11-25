
from stride import *
import sys


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
    x = sys.argv[2]
    arch = 3.14159 * 400e-3 / num_of_trans
    problem = Problem(name=name,
                      space=space, time=time)

    # Create medium
    if x == 'homo':
    	vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
    	vp.fill(1540.)
    	problem.medium.add(vp)
    elif x == 'NN':
        pred_rect = np.load('data/pred_rect.npy')
        pred_rect = pred_rect.T
        pred_rect = np.where(pred_rect == 0, 1540, pred_rect)
        pred_rect = np.where(pred_rect < 2000, 1540, pred_rect)
        vp = ScalarField(name='vp', grid=problem.grid, data=pred_rect, needs_grad=True)
        problem.medium.add(vp)
        problem.plot()
        vp.extended_plot()
    elif x == 'homo2':
        homo_rect = np.load('data/homo.npy')
        vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
        vp.data[...] = homo_rect
        vp.pad()
        problem.medium.add(vp)
        problem.plot()
        vp.extended_plot()
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

        if num_of_trans == 128:
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
    problem.plot()
    # Create acquisitions
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

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
                      f_max=freq, max_freqs=max_freqs)


if __name__ == '__main__':
    mosaic.run(main)


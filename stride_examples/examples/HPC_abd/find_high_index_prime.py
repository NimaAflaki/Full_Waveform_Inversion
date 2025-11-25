import numpy as np
import sys
import matplotlib.pyplot as plt
from stride import Space, Time, Problem, ScalarField
from stride.utils import fetch


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
    problem = Problem(name="cleaning", space=space, time=time)
    indices = np.load('high_index.npy')

    vp_model = sys.argv[1]
    file = vp_model + '_vp.h5'
    vp = ScalarField(name='vp', grid=problem.grid)
    fetch(vp_model, dest=file)
    vp.load(file)

    data = np.array(vp.data)
    index_set = set(map(tuple, indices))
    for i, j in index_set:
        data[i, j] = np.nan
    #np.save('clean_vp/' + vp_model + '_clean.npy', vp)
    frontier = []
    flag = False
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                    frontier.append((i, j))
                    flag = True
                    break
        if flag == True:
            break
    print('Initial frontier is: ', frontier)
    
    indices_prime = []
    while frontier:
        index = frontier.pop()
        i, j = index[0], index[1]
        indices_prime.append(index)
        if not np.isnan(data[i-1, j]) and (i-1, j) not in indices_prime:
            frontier.append((i-1, j))
        if not np.isnan(data[i+1, j]) and (i+1, j) not in indices_prime:
            frontier.append((i+1, j))
        if not np.isnan(data[i, j-1]) and (i, j-1) not in indices_prime:
            frontier.append((i, j-1))
        if not np.isnan(data[i, j+1]) and (i, j+1) not in indices_prime:
            frontier.append((i, j+1))

    
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('vp with indices_prime overlay')

    # Unzip the (i,j) tuples to x and y lists for scatter plot
    ys, xs = zip(*indices_prime)  # row -> y, col -> x

    # Plot the indices in red dots
    plt.scatter(xs, ys, color='red', s=1, label='indices_prime')

    plt.legend()
    plt.show()

    

    final = np.array(indices_prime)
    np.save('prime_index', final)

         

         
         
         

    print(np.nanmax(data))
    plt.imshow(data, cmap='gray', interpolation='nearest', vmax=1570)
    plt.colorbar()
    plt.title(vp_model) 
    plt.show()




if __name__ == '__main__':
        main()









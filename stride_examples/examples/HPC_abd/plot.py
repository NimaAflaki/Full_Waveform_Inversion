import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    name = sys.argv[1]
    data = np.load(name)
    print("    shape = ", data.shape)
    print("    size  = ", data.size)
    print('Max = ', np.max(data), 'Min = ', np.min(data), 'Avg = ', np.mean(data))
    plt.imshow(data, cmap='gray', interpolation='nearest', vmax=100)
    plt.colorbar()
    plt.title('NN and homo Difference')
    plt.show()


if __name__ == '__main__':
    main()

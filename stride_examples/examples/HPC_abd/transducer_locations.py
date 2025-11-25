import numpy as np

x_mm_transducer = np.load('x_mm_transducer.npy')
z_mm_transducer = np.load('z_mm_transducer.npy')

x_mm_transducer = x_mm_transducer + 210
z_mm_transducer = z_mm_transducer + 105

trans_coord = np.array(list(zip(x_mm_transducer, z_mm_transducer)))

np.save('transducer_locations.npy', trans_coord)

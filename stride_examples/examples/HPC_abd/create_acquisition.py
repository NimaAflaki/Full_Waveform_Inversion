import h5py
import sys
import numpy as np

def main():
    # Add Source Wavelet
    data = np.load('./data/FSA_decimated_0014.npy')
    new_data = np.load("my_complex_data.npy").astype(np.complex64)  # complex data
    new_extended_shape = np.array(new_data.shape, dtype=np.int64)
    new_inner = np.array([1, 2, 3])  # <-- replace with your actual value
    new_shape = np.array(new_data.shape, dtype=np.int64)
    new_step_size = np.array(10, dtype=np.int64)  # step size = 10
    file = sys.argv[1] + '-Acquisitions.h5'
    data = sys.argv[2] + '.npy'
    wavelet = np.load(data)
    f = h5py.File(file, 'r+')
    shots = f['shots'].keys()
    for shot in shots:
        f['shots'][shot]['wavelets']['data'][...] = wavelet
        #Update obs
        obs = f['shots'][shot]['observed']
        def replace_dataset(group, name, data, dtype=None):
            if name in group:
                del group[name]
            if dtype is not None:
                group.create_dataset(name, data=data, dtype=dtype)
            else:
                group.create_dataset(name, data=data)

    # --- Replace datasets ---
        replace_dataset(obs, "data", new_data, dtype=np.complex64)
        replace_dataset(obs, "dtype", np.string_("complex64"))
        replace_dataset(obs, "step_size", new_step_size)
        replace_dataset(obs, "shape", new_shape)
        replace_dataset(obs, "extended_shape", new_extended_shape)
        replace_dataset(obs, "inner", new_inner)
def replace_dataset(group, name, data, dtype=None):
    if name in group:
        del group[name]
    if dtype is not None:
        group.create_dataset(name, data=data, dtype=dtype)
    else:
        group.create_dataset(name, data=data)

if __name__ == '__main__':
    main()

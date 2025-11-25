import h5py
import sys
import numpy as np

def replace_dataset(group, name, data, dtype=None):
    """Helper function to replace a dataset in an HDF5 group"""
    original_attrs = {}
    if name in group:
        original_attrs = dict(group[name].attrs)
        del group[name]
    if dtype is not None:
        new_dataset = group.create_dataset(name, data=data, dtype=dtype)
    else:
        new_dataset = group.create_dataset(name, data=data)

    if 'is_ndarray' not in original_attrs:
        # Set default attributes for new datasets
        if isinstance(data, np.ndarray):
            new_dataset.attrs['is_ndarray'] = True
        else:
            new_dataset.attrs['is_ndarray'] = False

    # Restore all original attributes
    for attr_name, attr_value in original_attrs.items():
        new_dataset.attrs[attr_name] = attr_value

def main():
    # Load the FSA decimated data (shape: 64, 64, 1244)
    fsa_data = np.load('./data/FSA_decimated_0014.npy')

    # Get command line arguments
    file = sys.argv[1] + '-Acquisitions.h5'

    # Load wavelet data
    wavelet = np.load('wavelet.npy')

    # Open HDF5 file
    f = h5py.File(file, 'r+')
    shots = list(f['shots'].keys())

    # Process each shot
    for i, shot in enumerate(shots):
        # Update wavelet
        f['shots'][shot]['wavelets']['data'][...] = wavelet

        # Get the observed data group
        obs = f['shots'][shot]['observed']

        # Extract data for this shot (shape: 64, 1244)
        # Each shot corresponds to one transmit event
        new_data = fsa_data[i, :, :]  # Shape: (64 channels, 1244 samples)

        # Define new parameters
        new_shape = np.array([64, 1244], dtype=np.int64)
        new_extended_shape = np.array([64, 1244], dtype=np.int64)
        new_step_size = np.array(10, dtype=np.int64)

        # Update inner - keeping structure but changing 12440 to 1244
        # Based on your example: [[b'0', b'None', b'None'], [b'0', b'4200', b'None']]
        # But you mentioned 12440 needs to change to 1244
        # Assuming the structure might be [[b'0', b'None', b'None'], [b'0', b'1244', b'None']]
        new_inner = np.array([[b'0', b'None', b'None'],
                              [b'0', b'1244', b'None']], dtype=object)

        # Replace all datasets
        replace_dataset(obs, "data", new_data, dtype=np.complex64)
        replace_dataset(obs, "dtype", np.bytes_("complex64"))
        replace_dataset(obs, "step_size", new_step_size)
        replace_dataset(obs, "shape", new_shape)
        replace_dataset(obs, "extended_shape", new_extended_shape)
        replace_dataset(obs, "inner", new_inner)

    # Close the file
    f.close()
    print(f"Successfully updated {len(shots)} shots")

if __name__ == '__main__':
    main()

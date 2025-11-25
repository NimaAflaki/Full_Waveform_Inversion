import numpy as np
import h5py
import matplotlib.pyplot as plt

gt = np.load('data/gt_rect.npy').T
NN_dir = 'JHP_NN_Vp'
Homo_dir = 'JHP_Homo_Vp'
mask = np.load('mask.npy')
mask2 = np.load('mask2.npy')
NN_guess = np.load('data/pred_rect.npy').T
NN_guess = np.where(NN_guess == 0, 2500, NN_guess)
mask_NN = NN_guess < 2000
mask3 = mask & mask_NN

# 32x1637x1228 1st dim is time, 2&3 are data
NN = []
Homo = []

# Get NN/Homo data by time
for i in range(1, 33):
    if i < 10:
        file = NN_dir + '/JHP-Vp-0000' + str(i) + '.h5'
        f = h5py.File(file, 'r')
        NN.append(f['data'][()] * mask3)
        f.close()
        file = Homo_dir + '/JHP-Vp-0000' + str(i) + '.h5'
        f = h5py.File(file, 'r')
        Homo.append(f['data'][()] * mask3)
        f.close()
    elif i > 9:
        file = NN_dir + '/JHP-Vp-000' + str(i) + '.h5'
        f = h5py.File(file, 'r')
        NN.append(f['data'][()] * mask3)
        f.close()
        file = Homo_dir + '/JHP-Vp-000' + str(i) + '.h5'
        f = h5py.File(file, 'r')
        Homo.append(f['data'][()] * mask3)
        f.close()


# Apply mask
gt = gt * mask3


# Calculate avg error
Homo_error = []
NN_error = []
for NNplot, Homoplot in zip(NN, Homo):
    Homoplot_diff = np.array(gt - Homoplot)
    Homoplot_error = np.mean(np.abs(Homoplot_diff))
    Homo_error.append(Homoplot_error)
    NNplot_diff = np.array(gt - NNplot)
    NNplot_error = np.mean(np.abs(NNplot_diff))
    NN_error.append(NNplot_error)

error_diff = np.array(Homo_error) - np.array(NN_error)

# Plot error
x = range(1, 33)
plt.figure()
plt.plot(x, Homo_error, 'k-', label='Homo Error')
plt.plot(x, NN_error, 'b-', label='NN Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Avg Error (m/s)')
plt.legend()

plt.figure(2)
plt.plot(x, error_diff)
plt.ylim(-5, 5)
plt.title('Homo Error - NN Error (Positive Value = NN Better)')
plt.xlabel('Number of Iterations')
plt.ylabel('Error Difference (m/s)')

plt.show()














from stride import *
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    vp_NN = np.load('NN14.npy')
    vp_homo = np.load('Homo14.npy')
    gt = np.load('data/gt_rect.npy').T
    # Take Difference
    diff_homo = vp_homo - gt
    diff_NN = vp_NN - gt

    # Normalize to GT
    norm_diff_homo = diff_homo / gt
    norm_diff_NN = diff_NN / gt

    # Take absolute value
    abs_norm_diff_homo = np.abs(norm_diff_homo)
    abs_norm_diff_NN = np.abs(norm_diff_NN)

    # Show diff between homo and NN
    diff = abs_norm_diff_homo - abs_norm_diff_NN

    # Plot
    plt.imshow(gt)
    plt.colorbar()
    plt.show()

    plt.imshow(vp_homo)
    plt.colorbar()
    plt.show()



    plt.imshow(norm_diff_homo, vmax=1, vmin=-1)
    plt.colorbar(label='Normalized Difference')
    plt.show()

    plt.imshow(norm_diff_NN, vmax=1, vmin=-1)
    plt.colorbar(label='Normalized Difference')
    plt.show()

    plt.imshow(diff, cmap='coolwarm', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Difference')
    plt.show()

if __name__ == "__main__":
    main()

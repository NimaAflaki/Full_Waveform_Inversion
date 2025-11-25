import numpy as np

def main():
    # Load the arrays
    gt_rect = np.load("gt_rect.npy")
    pred_rect = np.load("pred_rect.npy")
    homo = np.load("homo.npy")
    # Print their shapes and sizes
    print("gt_rect:")
    print("  shape =", gt_rect.shape)
    print("  size  =", gt_rect.size)

    print("\npred_rect:")
    print("  shape =", pred_rect.shape)
    print("  size  =", pred_rect.size)

    print("\nHomo:")
    print("  shape =", homo.shape)
    print("  size  =", homo.size)


if __name__ == "__main__":
    main()

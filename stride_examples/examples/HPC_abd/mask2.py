from stride import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load GT
    gt = np.load('data/gt_rect.npy').T

    # Option 1, logical index
    bone = gt > 1700
    air = gt == 0
    mask1 = bone | air

    # Option 2: search
    mask_idx = set()
    frontier = []

    a, b = 1, 0
    while True:
        if gt[a, b] != 0:
            break
        b += 1
    print(gt[a,b])

    frontier.append((a, b))
    mask_idx.add((a, b))

    while frontier:
        i, j = frontier.pop()
        if gt[(i, j)] != 0 and gt[(i, j)] < 1700:
            mask_idx.add((i, j))
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for ele in neighbors:
                ni, nj = ele
                if ni < 0 or nj < 0 or ni>=gt.shape[0] or nj>=gt.shape[1]:
                    continue
                if ele in mask_idx:
                    continue
                frontier.append(ele)

    mask = np.zeros_like(gt, dtype=bool)
    for i, j in mask_idx:
        mask[i, j] = True

    mask[:, 0:600] = 0

    np.save('mask2.npy', mask)

    # Plot
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np


NN_guess = np.load('data/pred_rect.npy').T

mask = NN_guess > 2000



pred_rect = np.where(pred_rect == 0, 1540, pred_rect)
pred_rect = np.where(pred_rect < 2000, 1540, pred_rect)

pred_rect = np.where(pred_rect < 2000, 1540, pred_rect)

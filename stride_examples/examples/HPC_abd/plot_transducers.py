import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap

xr_mm = np.load('xr_mm.npy')
zr_mm = np.load('zr_mm.npy')
x_mm_transducer = np.load('x_mm_transducer.npy')
z_mm_transducer = np.load('z_mm_transducer.npy')
x_mm_kwave = np.load('x_mm_kwave.npy')
z_mm_kwave = np.load('z_mm_kwave.npy')


#gt_kwave = np.load('gt_kwave.npy')
#gt_rect = np.load('gt_rect.npy')
pred_rect= np.load('pred_rect.npy')

colors = ['black', 'red', 'orange', 'yellow', 'white']
custom_cmap = LinearSegmentedColormap.from_list('sos_custom', colors, N=256)

Xr, Zr = np.meshgrid(xr_mm, zr_mm)
extent = [x_mm_kwave[0], x_mm_kwave[-1], z_mm_kwave[-1], z_mm_kwave[0]]

plt.figure(figsize=(12, 7))
#plt.imshow(
 #   gt_kwave, extent=extent, cmap=custom_cmap,
  #  vmin=1478, vmax=2425, aspect='auto'
#)
#plt.imshow(
 #   gt_rect, extent=extent, cmap=custom_cmap,
 #   vmin=1478, vmax=2425, aspect='auto'
#)
plt.imshow(
    pred_rect, extent=extent, cmap=custom_cmap,
    vmin=1478, vmax=2425, aspect='auto'
)


plt.colorbar(label='Speed of Sound (m/s)')

plt.scatter(x_mm_transducer, z_mm_transducer, color='red', s=30, label='Transducers')
plt.plot(
    [x_mm_kwave[0], x_mm_kwave[-1], x_mm_kwave[-1], x_mm_kwave[0], x_mm_kwave[0]],
    [z_mm_kwave[0], z_mm_kwave[0], z_mm_kwave[-1], z_mm_kwave[-1], z_mm_kwave[0]],
    color='lime', linewidth=1.5, linestyle='--', label='k-Wave Region'
)

plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.title('SOS Map (Black to White Gradient)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

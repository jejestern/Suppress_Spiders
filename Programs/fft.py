#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:44:30 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/rphi_plane_spline3_R150_R300.txt").reshape(150, 2356)

fig, ax = plt.subplots(1,1)
im = ax.imshow(data, origin='lower', aspect='auto', vmin=0, vmax= 5, 
               extent=[0, 360, 150, 300])
plt.tight_layout()
plt.colorbar(im)
plt.show()

fourier = np.fft.fftshift(np.fft.fft2(data))

plt.figure()
plt.imshow(np.log(abs(fourier)), cmap='gray', aspect='auto')
plt.tight_layout()
plt.colorbar()
plt.show()

# R150-300 usual length
#fourier[65:85, -706:-550] = 1
#fourier[65:85, 550:706] = 1

# R150-300 1000x1000
#fourier[400:600, -600:600] = 1

# R300-450 usual length
fourier[65:85, -1177:-800] = 1
fourier[65:85, 800:1178] = 1

"""
x_len, y_len = fourier.shape
x_center = x_len/2 - 1
y_center = y_len/2 - 1
# Define the corresponding polar coordinates to the x-y coordinates
r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
mask_r = (r_array <= 50)
mask_2 = (r_array >= 60)

fourier = fourier*(mask_2 + mask_r)
"""
plt.figure()
plt.imshow(np.log(abs(fourier)), cmap='gray', aspect='auto')
plt.tight_layout()
plt.colorbar()
plt.show()

img_back = abs(np.fft.ifft2(fourier))
fig, ax = plt.subplots(1,1)
im = ax.imshow(img_back, origin='lower', aspect='auto', vmin=0, vmax= 5, 
               extent=[0, 360, 150, 300])
plt.tight_layout()
plt.colorbar(im)
plt.show()

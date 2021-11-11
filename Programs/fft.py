#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:44:30 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/rphi_plane_spline3_R150_R300.txt").reshape(150, 1413)

fig, ax = plt.subplots(1,1)
im = ax.imshow(data, origin='lower', aspect='auto', vmin=0, vmax= 20, 
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

#fourier[-70: , 700:710] = 1
#fourier[:70, 700:710] = 1

fourier[65:85, -700:] = 1
fourier[65:85, :700] = 1

plt.figure()
plt.imshow(np.log(abs(fourier)), cmap='gray', aspect='auto')
plt.tight_layout()
plt.colorbar()
plt.show()

img_back = abs(np.fft.ifft2(fourier))
fig, ax = plt.subplots(1,1)
im = ax.imshow(img_back, origin='lower', aspect='auto', vmin=0, vmax= 20, 
               extent=[0, 360, 150, 300])
plt.tight_layout()
plt.colorbar(im)
plt.show()

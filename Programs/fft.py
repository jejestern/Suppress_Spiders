#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:44:30 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/rphi_plane_spline3_R150_R300.txt").reshape(150, 1413)

fig, ax = plt.subplots()
        
ax.imshow(data, origin='lower', aspect='auto')
plt.tight_layout()

#plt.plot(rphi_grid[:,0], rphi_grid[:,1], 'b.', ms=1)
plt.show()


fourier = np.fft.fftshift(np.fft.fft2(data))
print(fourier)
plt.figure()
plt.imshow(np.log(abs(fourier)), cmap='gray', aspect='auto')
plt.tight_layout()
plt.show()


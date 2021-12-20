#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:21:19 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt


def fourier(image, title):
    
    fourier_im = np.fft.fftshift(np.fft.fft2(image))

    # Plot the output
    plt.figure(figsize=(8,3.5))

    plt.subplot(121)
    plt.imshow(image, origin='lower', cmap='gray', vmin=0, vmax=1.0)
    plt.title(r'Image')
    plt.colorbar()
    
    epsilon = 10**(-6) # In order to be able to take the log
    
    plt.subplot(122)
    plt.imshow(np.log(abs(fourier_im)+epsilon), origin='lower', cmap='gray', vmin=-1, vmax=5.0)
    plt.title(r'Fourier Transformed Image')
    plt.colorbar()
        
    plt.tight_layout()
    plt.savefig("fourier/fft_simulation"+ title +".pdf")
    plt.show()
    
    return fourier_im


sim_1 = np.zeros((100, 100))
sim_1[:, 10] = 1
fourier(sim_1, "oneline")

sim_2 = np.zeros((100, 100))
i = 20
while i < 100:
    sim_2[:, i] = 1
    i += 20
fourier(sim_2, "morelines")

sim_1 = np.zeros((100, 100))
sim_1[:, 6:14] = 1
fourier(sim_1, "onebeam")

sim_2 = np.zeros((100, 100))
i = 20
while i < 100:
    sim_2[:, i-4:i+4] = 1
    i += 20
fourier(sim_2, "morebeams")

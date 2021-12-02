#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:21:19 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt


def fourier(image):
    
    fourier_im = np.fft.fftshift(np.fft.fft2(image))

    # Plot the output
    plt.figure(figsize=(28,12))

    plt.subplot(121)
    plt.imshow(image, origin='lower', cmap='gray', aspect='auto')
    plt.title(r'Image')
    plt.colorbar()
    
    epsilon = 10**(-6) # In order to be able to take the log
    
    plt.subplot(122)
    plt.imshow(np.log(abs(fourier_im)+epsilon), origin='lower', cmap='gray', aspect='auto')
    plt.title(r'Fourier Transformed Image')
    plt.colorbar()
        
    plt.tight_layout()
    plt.show()
    
    return fourier_im

sim_1 = np.zeros((1000, 1000))
sim_1[:, 200] = 1
fourier(sim_1)

sim_2 = np.zeros((1000, 1000))
i = 200
while i < 1000:
    sim_2[:, i] = 1
    i += 200
fourier(sim_2)

sim_1 = np.zeros((1000, 1000))
sim_1[:, 180:220] = 1
fourier(sim_1)

sim_2 = np.zeros((1000, 1000))
i = 200
while i < 1000:
    sim_2[:, i-20:i+20] = 1
    i += 200
fourier(sim_2)

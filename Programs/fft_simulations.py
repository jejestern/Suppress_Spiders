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
    plt.imshow(sim, origin='lower', cmap='gray', aspect='auto')
    plt.title(r'Image')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(abs(fourier_im), origin='lower', cmap='gray', aspect='auto')
    plt.title(r'Fourier Transformed Image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    return fourier_im

sim = np.zeros((1000, 1000))
sim[:, 200] = 1
fourier(sim)


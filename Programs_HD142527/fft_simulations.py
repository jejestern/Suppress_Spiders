#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:21:19 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def fourier(image, title):
    
    fourier_im = np.fft.fftshift(np.fft.fft2(image))

    # Frequency plane
    freq = np.fft.fftfreq(image.shape[0])
    freq = np.fft.fftshift(freq)

    # Plot the output
    plt.figure(figsize=(8,3.5))

    plt.subplot(121)
    plt.imshow(image, origin='lower', cmap='gray', vmin=0, vmax=1.0)
    plt.title(r'Image')
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.colorbar()
    
    epsilon = 10**(-6) # In order to be able to take the log
    
    plt.subplot(122)
    plt.imshow(abs(fourier_im+epsilon), origin='lower', cmap='gray', norm=LogNorm(vmin=1),
               extent=[freq[0], freq[-1], freq[0], freq[-1]])
    plt.title(r'Fourier Transformed Image')
    plt.xlabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
    plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
    plt.colorbar()
        
    plt.tight_layout()
    plt.savefig("fourier/fft_simulation"+ title +".pdf")
    plt.show()
    
    # Plot the horizontal cut thorugh the FFT
    plt.figure()
    plt.semilogy(freq, abs(fourier_im[int(image.shape[0]/2), :] + 0.0001))
    #plt.ylim((10**(0), 10**(3)))
    plt.title("FFT horizontal cut at freq 0")
    #plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
    #plt.legend()
    plt.tight_layout()
    plt.savefig("fourier/fft_simulation_cut"+ title +".pdf")
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

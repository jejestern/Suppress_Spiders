#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:05:00 2021

@author: jeje
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm


def e_func(x, a, b, c):

    return a * np.exp(-b * x) + c


shape = 150, 1413
R_1, R_2 = 150, 300

data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs_HD142527/rphi_plane_spline3.txt").reshape(shape)

# Fourier transform
fourier = np.fft.fftshift(np.fft.fft2(data))

# Plot the output
aspect_value = 360/shape[1]
plt.figure(figsize=(8/aspect_value, 8))

plt.subplot(211)
plt.imshow(data, origin='lower', aspect=aspect_value, vmin=0, vmax= 20, 
               extent=[0, 360, R_1, R_2])
plt.title(r'Image in r-phi plane')
plt.colorbar()
    
plt.subplot(212)
plt.imshow(abs(fourier), origin='lower', cmap='gray', norm=LogNorm(vmin=1), aspect=1)
plt.title(r'Fourier Transformed Image')
plt.colorbar()
        
plt.tight_layout()
plt.show()


# Sum up the image along the phi axis
r_trend = np.sum(data, axis = 1)/shape[1]
radi = np.arange(shape[0])

## Fitting an exponential
popt, pcov = curve_fit(e_func, radi, r_trend)

plt.figure()
plt.plot(radi, e_func(radi, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.plot(radi, r_trend, 'b.', label="intensity distribution of longest speckle")
plt.legend()
plt.show()

for i in range(shape[1]):
    data[:,i] = data[:,i] - e_func(radi, *popt)
    
# Plot the output
fourier = np.fft.fftshift(np.fft.fft2(data))

plt.figure(figsize=(8/aspect_value, 8))

plt.subplot(211)
plt.imshow(data, origin='lower', aspect=aspect_value, vmin=0, vmax= 4, 
               extent=[0, 360, R_1, R_2])
plt.title(r'Image in r-phi plane flatten')
plt.colorbar()
    
epsilon = 10**(-6) # In order to be able to take the log even if there are zeros in the array
    
plt.subplot(212)
plt.imshow(abs(fourier), origin='lower', cmap='gray', norm=LogNorm(vmin=1), aspect=1)
plt.title(r'Fourier Transformed Image')
plt.colorbar()
        
plt.tight_layout()
plt.show()
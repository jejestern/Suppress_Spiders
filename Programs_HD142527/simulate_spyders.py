#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:21:00 2022

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformations_functions import to_rphi_plane

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianBeam(base, x_position, D0):
    rows, cols = base.shape
    for x in range(cols):
        base[:,x] = np.exp(((-distance((0,x), (0, x_position))**2)/(2*(D0**2))))
    return base


# Create an image with only zeros with the same shape as the star images have
x_len, y_len = 1024, 1024
x_center = x_len/2 - 1
y_center = y_len/2 - 1

zeros = np.zeros((x_len, y_len))

# Choose the radial range into which we are warping the image
R_1 = 254
R_2 = 454
Imax = 1.0
"""        
# Plot the created image  
plt.imshow(zeros, origin='lower', cmap='gray', vmin=0, vmax=Imax)
plt.colorbar()
plt.tight_layout()
plt.show()
"""
# Warp the image into the r-phi plane
warp = to_rphi_plane(zeros, (x_len, y_len), R_1, R_2)
warp_or = warp.T
warp_shape = warp.shape

aspect_value = (360/warp_shape[0])/((R_2-R_1)/warp_shape[1])

# We insert beams at the positions of the spyders
beams = warp_or.copy()
beams[:, 31:53] = 1
beams[:, 654:686] = 1
beams[:, 1143:1167] = 1
beams[:, 1767:1791] = 1
        
# Fourier transform the warped image with beams
fft_beams = np.fft.fftshift(np.fft.fft2(beams))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(beams, origin='lower', aspect=aspect_value, vmin=0, vmax=Imax, 
           extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_beams), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
           aspect=aspect_value, extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()

# We insert smoothed (gaussian) beams at the positions of the spyders
beamG1 = gaussianBeam(warp_or.copy(), 42, 11)
beamG2 = gaussianBeam(warp_or.copy(), 670, 16)
beamG3 = gaussianBeam(warp_or.copy(), 1155, 12)
beamG4 = gaussianBeam(warp_or.copy(), 1779, 12)

beamG = beamG1 + beamG2 + beamG3 + beamG4
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(beamG, origin='lower', aspect=aspect_value, vmin=0, vmax=Imax, 
           extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_beamG), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
           aspect=aspect_value, extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()
        

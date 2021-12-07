#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:56:04 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
from transformations_functions import polar_corrdinates_grid, radius_mask, angle_mask, to_rphi_plane
from astropy.visualization import simple_norm
#from aperture_radius import aperture_phot

x_len, y_len = 1000, 1000

circle = np.ones((x_len, y_len))

# Define the circle
circle_center = (300, 511)
circle_radius = (0, 10)
r_array, phi_array = polar_corrdinates_grid((x_len, y_len), circle_center)
mask_r = radius_mask(r_array, circle_radius)
mask_phi = angle_mask(phi_array, (0, 2*np.pi))
mask = mask_r & mask_phi

circle = circle*mask
        
plt.imshow(circle, origin='lower', cmap='gray')
plt.colorbar()
plt.show()

# Start with the transformation to r-phi plane
## Choose the radial range
R_1 = 150
R_2 = 300

warped = to_rphi_plane(circle, (x_len, y_len), R_1, R_2)
warped_or = warped.T
warped_shape = warped.shape

aspect_value = (360/warped_shape[0])/((R_2-R_1)/warped_shape[1])
"""
fig, ax = plt.subplots(1,1, figsize=(8/aspect_value, 8))
im = ax.imshow(warped_or, origin='lower', aspect=aspect_value, vmin=0, vmax= 1, 
               extent=[0, 360, R_1, R_2])
plt.tight_layout()
plt.colorbar(im)
plt.show()
"""
    
fourier_w = np.fft.fftshift(np.fft.fft2(warped_or))

# Plot the output
plt.figure(figsize=(8/aspect_value, 8))

plt.subplot(211)
plt.imshow(warped_or, origin='lower', cmap='gray', aspect=aspect_value, 
           extent=[0, 360, R_1, R_2])
plt.title(r'Image in r-phi plane')
plt.colorbar()
    
norm_fft = simple_norm(abs(fourier_w), 'log', percent=99.9) # In order to be able to take the log even if there are zeros in the array
    
plt.subplot(212)
plt.imshow(abs(fourier_w), origin='lower', cmap='gray', norm=norm_fft, aspect=1)
plt.title(r'Fourier Transformed Image')
plt.colorbar()
        
plt.tight_layout()
plt.show()
"""
# Define the norm for afterwards plotting the image
norm = simple_norm(circle, 'log', percent=99.9)
aperture_rad = 11.0        

fmean, sigma, SN, ratio = aperture_phot(circle, norm, circle_center, [(511,511)], aperture_rad, True)
"""
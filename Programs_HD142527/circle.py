#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:56:04 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
from transformations_functions import polar_corrdinates_grid, radius_mask, angle_mask, to_rphi_plane, from_rphi_plane
from matplotlib.colors import LogNorm
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture, EllipticalAnnulus, aperture_photometry 
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

# Calculate the aperture sum of the circle 
ap_im = CircularAperture([511,300], r=circle_radius[1])
annu_im = CircularAnnulus([511,300], r_in=15, r_out=20)
phot_table_im = aperture_photometry(circle, ap_im)
aperture_im = np.array(phot_table_im['aperture_sum'])
annu_masks_im = annu_im.to_mask(method='center') 
annu_data_im = annu_masks_im.multiply(circle)
mask_im = annu_masks_im.data
annu_data_1d_im = annu_data_im[mask_im > 0]
spixel_im = sum(annu_data_1d_im)/annu_data_1d_im.shape[0]
f_ap_im = aperture_im - ap_im.area*spixel_im

# Plot the image        
plt.imshow(circle, origin='lower', cmap='gray')
ap_im.plot(color ='r', lw=1.0)
annu_im.plot(color ='#0547f9', lw=1.0)
plt.colorbar()
plt.savefig("interpolation/Circle_image.pdf")
plt.show()

print("The aperture of the circle is: ", f_ap_im[0])

# Start with the transformation to r-phi plane
## Choose the radial range
R_1 = 100
R_2 = 300

# Warp the image to the rphi plane
warped = to_rphi_plane(circle, (x_len, y_len), R_1, R_2)
warped_or = warped.T
warped_shape = warped.shape

aspect_value = (360/warped_shape[0])/((R_2-R_1)/warped_shape[1])

# Calculate the aperture sum of the circle in the warped image
ap_w = CircularAperture([640,99], 10)
ap_w_draw = EllipticalAperture([183.5,200], 10*aspect_value, 10)
annu_w = CircularAnnulus([640,99], 15, 20)
annu_w_draw = EllipticalAnnulus([183.5,200], 15*aspect_value, 20*aspect_value, 20, 15)
phot_table_w = aperture_photometry(warped_or, ap_w)
aperture_w = np.array(phot_table_w['aperture_sum'])
print(aperture_w)
annu_masks_w = annu_w.to_mask(method='center') 
annu_data_w = annu_masks_w.multiply(warped_or)
mask_w = annu_masks_w.data
annu_data_1d_w = annu_data_w[mask_w > 0]
spixel_w = sum(annu_data_1d_w)/annu_data_1d_w.shape[0]
f_ap_w = aperture_w - ap_w.area*spixel_w

    
fourier_w = np.fft.fftshift(np.fft.fft2(warped_or))

# Plot the output
plt.figure(figsize=(8, 8*aspect_value))

#plt.subplot(211)
plt.imshow(warped_or, origin='lower', cmap='gray', aspect=aspect_value, 
           extent=[0, 360, R_1, R_2], vmin=0, vmax=1.0)
#plt.imshow(warped_or, origin='lower', cmap='gray', aspect=aspect_value, 
 #           vmin=0, vmax=1.0)
ap_w_draw.plot(color ='r', lw=1.0)
annu_w_draw.plot(color ='#0547f9', lw=1.0)
#plt.title(r'Image in r-phi plane')
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()
"""  
plt.subplot(212)
plt.imshow(abs(fourier_w), origin='lower', cmap='gray', norm=LogNorm(vmin=1), aspect=1)
plt.title(r'Fourier Transformed Image')
plt.colorbar()
"""        
plt.tight_layout()
plt.savefig("interpolation/Circle_center.pdf")
plt.show()

print("The aperture of the circle in the warped image is: ", f_ap_w[0])

"""
# Define the norm for afterwards plotting the image
norm = simple_norm(circle, 'log', percent=99.9)
aperture_rad = 11.0        

fmean, sigma, SN, ratio = aperture_phot(circle, norm, circle_center, [(511,511)], aperture_rad, True)
"""

# Back transformation
h2 = from_rphi_plane(warped, (x_len, y_len), R_1, R_2)

# Calculate the aperture sum of the circle 
ap_im = CircularAperture([511,300], r=circle_radius[1])
annu_im = CircularAnnulus([511,300], r_in=15, r_out=20)
phot_table_im = aperture_photometry(circle, ap_im)
aperture_im = np.array(phot_table_im['aperture_sum'])
annu_masks_im = annu_im.to_mask(method='center') 
annu_data_im = annu_masks_im.multiply(circle)
mask_im = annu_masks_im.data
annu_data_1d_im = annu_data_im[mask_im > 0]
spixel_im = sum(annu_data_1d_im)/annu_data_1d_im.shape[0]
f_ap_im = aperture_im - ap_im.area*spixel_im

# Plot the image        
plt.imshow(circle, origin='lower', cmap='gray')
ap_im.plot(color ='r', lw=1.0)
annu_im.plot(color ='#0547f9', lw=1.0)
plt.imshow(h2, origin='lower', cmap='gray', vmin=0, vmax=1.0)
plt.colorbar()
plt.show()

print("The aperture of the circle after back transformation is: ", f_ap_im[0])
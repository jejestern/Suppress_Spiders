#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 05:21:19 2021

@author: jeje
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from transformations_functions import polar_corrdinates_grid, to_rphi_plane, radius_mask, angle_mask, from_rphi_plane
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture, EllipticalAnnulus, aperture_photometry 
#from scipy import interpolate


# This part takes the argument and saves the folder 
if not len(argv) == 1:
    print("Wrong number of arguments!")
    print("Usage: python ghosts.py")
    print("Exiting...")
    exit()

# The image path of the images taken in the P2 mode
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

# We define the positions of the ghosts
gh_pos = [(892.0, 598.0), (213.0, 387.0)]

for image_name in files[0:3]:
    if image_name.endswith("1.fits"): 
        # Reading in the images from camera 1
        img_data = fits.getdata(path + "/" + image_name, ext=0)
        fits.info(path + "/" + image_name)

        # Vertical flip image data to have the same convention as ds9
        #axis2fl=int(img_data.ndim-2)
        #print('axis to flip:',axis2fl)
        #img_ori = np.flip(img_data, axis2fl)

        # Choose the intensity 1
        int1 = img_data[0,:,:]
        
        x_len, y_len = int1.shape
        x_center = x_len/2 - 1
        y_center = y_len/2 - 1
        
        # Choose the radial range
        R_1 = 290
        R_2 = 490
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        # Calculate the aperture sum of the ghost 1
        ap_g = CircularAperture(gh_pos[0], r=6.0)
        annu_g = CircularAnnulus(gh_pos[0], r_in=10, r_out=15)
        phot_table_g = aperture_photometry(int1, ap_g)
        aperture_g = np.array(phot_table_g['aperture_sum'])
        annu_masks_g = annu_g.to_mask(method='center') 
        annu_data_g = annu_masks_g.multiply(int1)
        mask_g = annu_masks_g.data
        annu_data_1d_g = annu_data_g[mask_g > 0]
        spixel_g = sum(annu_data_1d_g)/annu_data_1d_g.shape[0]
        f_ap_g = aperture_g - ap_g.area*spixel_g

        print("The aperture of the ghost is: ",f_ap_g[0])
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=4)
        ap_g.plot(color ='r', lw=1.0)
        annu_g.plot(color ='#0547f9', lw=1.0)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
        
        
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_or = warped.T
        warped_shape = warped.shape
        
        aspect_value = (360/warped_shape[0])/((R_2-R_1)/warped_shape[1])
        
        # Calculate the aperture sum of the circle in the warped image
        ap_w = CircularAperture([1924,100], 6)
        ap_w_draw = EllipticalAperture([282.9,391], 6*aspect_value, 6)
        annu_w = CircularAnnulus([1924,100], 10, 15)
        annu_w_draw = EllipticalAnnulus([282.9,391], 10*aspect_value, 15*aspect_value, 15, 10)
        phot_table_w = aperture_photometry(warped_or, ap_w)
        aperture_w = np.array(phot_table_w['aperture_sum'])
        annu_masks_w = annu_w.to_mask(method='center') 
        annu_data_w = annu_masks_w.multiply(warped_or)
        mask_w = annu_masks_w.data
        annu_data_1d_w = annu_data_w[mask_w > 0]
        spixel_w = sum(annu_data_1d_w)/annu_data_1d_w.shape[0]
        f_ap_w = aperture_w - ap_w.area*spixel_w

        print("After the transformation the aperture is: ", f_ap_w[0])
        
        fig, ax = plt.subplots(1,1, figsize=(8, 8*aspect_value))
        im = ax.imshow(warped_or, origin='lower', aspect=aspect_value, vmin=0, 
                       vmax= 4, extent=[0, 360, R_1, R_2])
        #im = ax.imshow(warped_or, origin='lower', aspect=aspect_value, vmin=0, 
         #              vmax= 4)
        ap_w_draw.plot(color ='r', lw=1.0)
        annu_w_draw.plot(color ='#0547f9', lw=1.0)
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        

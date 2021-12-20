#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Include a model planet (copy of the star, just a lot fainter) and investigate
the effect on the FFT (as well as on the warping).

Created on 2021-12-20
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from transformations_functions import polar_corrdinates_grid, to_rphi_plane, radius_mask, angle_mask, from_rphi_plane
from matplotlib.colors import LogNorm
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
gh_pos = [(891.0, 600.0), (213.0, 387.0)]

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
        R_1 = 254
        R_2 = 454
        Imax = 5
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
              
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_or = warped.T
        warped_shape = warped.shape
        
        # Fourier transform the warped image
        fourier_w = np.fft.fftshift(np.fft.fft2(warped_or))
        
        aspect_value = (360/warped_shape[0])/((R_2-R_1)/warped_shape[1])
        
        # Plotting the warped and fft of it
        plt.figure(figsize=(8, 16*aspect_value))
        
        plt.subplot(211)
        plt.imshow(warped_or, origin='lower', aspect=aspect_value, vmin=0, 
                       vmax=Imax, extent=[0, 360, R_1, R_2])
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier_w), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
                   aspect=aspect_value, extent=[0, 360, R_1, R_2])
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar()
        
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        
        
        # We cut out the star and insert it a lot less bright in the top left 
        # part of the image
        intens = 10**(-5)
        int1[550:950, 50:450] += int1[int(y_center-200):int(y_center+200),
                                     int(x_center-200):int(x_center+200)]*intens 
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
              
        warped_model = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_m_or = warped_model.T
        
        ## Fourier transform the warped image (model star)
        fourier_m_w = np.fft.fftshift(np.fft.fft2(warped_m_or))
        
        ## Plotting the warped image and its fft (model star)
        plt.figure(figsize=(8, 16*aspect_value))
        
        plt.subplot(211)
        plt.imshow(warped_m_or, origin='lower', aspect=aspect_value, vmin=0, 
                       vmax=Imax, extent=[0, 360, R_1, R_2])
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier_m_w), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
                   aspect=aspect_value, extent=[0, 360, R_1, R_2])
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar()
        
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()

        # Plot and calculate the fft(model_star)-fft 
        fft_model = fourier_m_w - fourier_w
        plt.figure(figsize=(8, 8*aspect_value))
        plt.imshow(abs(fft_model), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
                   aspect=aspect_value, extent=[0, 360, R_1, R_2])
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        
"""      
        h2 = from_rphi_plane(warped, (x_len, y_len), R_1, R_2)
        plt.imshow(h2, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        plt.colorbar()
        plt.show()
        
        
        plt.imshow(h2-int1*mask, origin='lower', cmap='gray', vmin=-0.1, vmax=0.1)
        plt.colorbar()
        plt.show()  
"""
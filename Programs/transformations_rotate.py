#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:11:51 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from scipy.ndimage.interpolation import rotate 


def transform_to_rphi_scipy(image, R_start, R_end):
    
    # Define the shape and of the image (position of the star)
    x_len, y_len = image.shape
    center = x_len/2 - 1

    # Define the shape of the new coordinate system
    R_len = R_end - R_start
    phi_len = 360

    # Polar = [[], [], ...] where x-axis becomes phi and y-axis becomes radius
    polar = np.zeros((R_len, phi_len)) * np.nan

    degree = np.arange(phi_len/2)
    for deg in degree:
        rotated = rotate(int1, -deg, axes=(1,0))
            
        slice1 = rotated[int(center+R_start):int(center+R_end), int(center):int(center+1)][:,0]
        slice2 = rotated[int(center-R_end):int(center-R_start), int(center):int(center+1)][:,0]
        polar[:, int(deg)] = slice1
        slice2 = np.flip(slice2)
        polar[:, int(deg+180)] = slice2

    return polar

# The image path of the images taken in the P2 mode
path = "/home/jeje/Dokumente/Masterthesis/Programs/ZirkumstellareScheibe_HD142527/P2_mode"
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
        
        R_start = 150
        R_end = 300
        
        rotated = rotate(int1, -90, axes=(1,0))
        
        plt.imshow(int1, origin='lower', cmap='gray', vmin=0, vmax=100)
        plt.colorbar()
        plt.show()
        
        plt.imshow(rotated, origin='lower', cmap='gray', vmin=0, vmax=100)
        plt.colorbar()
        plt.show()
        
        transformed = transform_to_rphi_scipy(int1, R_start, R_end)
        
        plt.imshow(transformed, origin='lower', cmap='gray')
        plt.colorbar()
        plt.show()

        file1 = open("radtophi_rotate.txt", "w") 
        for row in transformed:
            np.savetxt(file1, row) 
        file1.close()
        

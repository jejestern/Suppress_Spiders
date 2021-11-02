#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different methods to transform an image into polar coordinates.
Atention: the x-axis is the vertical axis and the y-axis is the horizontal axis
in this case.

Created on 2021-11-02
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from transformations_1try import polar_corrdinates_grid
from scipy import interpolate
    

def func(x, y):

    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


# This part takes the argument and saves the folder 
if not len(argv) == 1:
    print("Wrong number of arguments!")
    print("Usage: python ghosts.py")
    print("Exiting...")
    exit()

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
        
        x_len, y_len = int1.shape
        x_center = x_len/2 - 1
        y_center = y_len/2 - 1
        
        # Choose the radial range
        R_1 = 150
        R_2 = 300
        
        r_grid, phi_grid = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        r_flat = r_grid.flatten()
        phi_flat = phi_grid.flatten()
        int1_flat = int1.flatten()
        
        print(len(r_flat))
        print(phi_flat)
        print(int1_flat.shape)
        #func = interpolate.interp2d(r_flat, phi_flat, int1, kind = 'cubic')

        grid_x, grid_y = np.mgrid[150:300:151j, 0:2*np.pi:1300j]
        
        rphi_grid = np.vstack((r_flat, phi_flat)).T

        grid_z2 = interpolate.griddata(rphi_grid, int1_flat, (grid_x, grid_y), method='linear')
                
        # Vertical flip image data to have the same convention as ds9
        axis2fl=int(grid_z2.ndim-2)
        #print('axis to flip:',axis2fl)
        grid_z2 = np.flip(grid_z2, axis2fl)
        
        
        fig, ax = plt.subplots()
        
        ax.imshow(grid_z2, aspect='auto')
        plt.tight_layout()

        #plt.plot(rphi_grid[:,0], rphi_grid[:,1], 'b.', ms=1)
        plt.show()

        rng = np.random.default_rng()

        points = rng.random((10, 2))

        values = func(points[:,0], points[:,1])

        
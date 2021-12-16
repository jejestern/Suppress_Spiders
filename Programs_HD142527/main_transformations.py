#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different methods to transform an image into polar coordinates.
Attention: the x-axis is the vertical axis and the y-axis is the horizontal axis
in this case.

Created on 2021-11-02
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from transformations_functions import polar_corrdinates_grid, to_rphi_plane, radius_mask, angle_mask, from_rphi_plane
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
        R_1 = 290
        R_2 = 490
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=3)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
        
        
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_or = warped.T
        warped_shape = warped.shape
        
        aspect_value = (360/warped_shape[0])/((R_2-R_1)/warped_shape[1])
        fig, ax = plt.subplots(1,1, figsize=(8, 8*aspect_value))
        im = ax.imshow(warped_or, origin='lower', aspect=aspect_value, vmin=0, 
                       vmax= 3, extent=[0, 360, R_1, R_2])
        plt.xlabel(r'$\varphi$ [degrees]')
        plt.ylabel('Radius')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        
        warped_file = open("rphi_plane_spline3_R150_R300.txt", "w") 
        for row in warped_or:
            np.savetxt(warped_file, row) 
        warped_file.close()
        print("The shape of the warped image is: ", warped.shape)
        
        h2 = from_rphi_plane(warped, (x_len, y_len), R_1, R_2)
        plt.imshow(h2, origin='lower', cmap='gray', vmin=0, vmax=3)
        plt.colorbar()
        plt.show()
        
        
        plt.imshow(h2-int1*mask, origin='lower', cmap='gray', vmin=-0.1, vmax=0.1)
        plt.colorbar()
        plt.show()
        
"""            
        r_grid, phi_grid = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        r_flat = r_grid.flatten()
        phi_flat = phi_grid.flatten()
        int1_flat = int1.flatten()
        
        print(len(r_flat))
        print(phi_flat)
        print(int1_flat.shape)
        #func = interpolate.interp2d(r_flat, phi_flat, int1, kind = 'cubic')

        grid_x, grid_y = np.mgrid[150:300:151j, 0:2*np.pi-0.01:1300j]
        
        rphi_grid = np.vstack((r_flat, phi_flat)).T
        
        print(grid_x)
        print(grid_y)

        grid_z2 = interpolate.griddata(rphi_grid, int1_flat, (grid_x, grid_y), method='linear')
                
        # Vertical flip image data to have the same convention as ds9
        axis2fl=int(grid_z2.ndim-2)
        #print('axis to flip:',axis2fl)
        grid_z2 = np.flip(grid_z2, axis2fl)
        
        print(grid_z2.shape)
        
        fig, ax = plt.subplots()
        
        ax.imshow(grid_z2, aspect='auto')
        plt.tight_layout()

        #plt.plot(rphi_grid[:,0], rphi_grid[:,1], 'b.', ms=1)
        plt.show()

        
        print(grid_z2)
        file1 = open("radtophi_interpolation.txt", "w") 
        for row in grid_z2:
            np.savetxt(file1, row) 
        file1.close()
"""    
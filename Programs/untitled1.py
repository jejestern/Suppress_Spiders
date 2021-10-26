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
        
        rotated = rotate(int1, 90, axes=(0,1))
        
        plt.imshow(int1, origin='lower', cmap='gray', vmin=0, vmax=100)
        plt.colorbar()
        plt.show()
        
        plt.imshow(rotated, origin='lower', cmap='gray', vmin=0, vmax=100)
        plt.colorbar()
        plt.show()
        

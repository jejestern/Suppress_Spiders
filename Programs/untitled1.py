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

radius = np.linspace(0, 1, 50)
angle = np.linspace(0, 2*np.pi, radius.size)
r_grid, a_grid = np.meshgrid(radius, angle)
data = np.sqrt((r_grid/radius.max())**2 + (a_grid/angle.max())**2)

fig, ax = plt.subplots()
ax.imshow(data, origin='lower')

def polar_to_cartesian(data):
    new = np.zeros_like(data) * np.nan
    x = np.linspace(-1, 1, new.shape[1])
    y = np.linspace(-1, 1, new.shape[0])
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            x0, y0 = x[j], y[i]
            r, a = np.sqrt(x0**2 + y0**2), np.arctan2(y0, x0)
            data_i = np.argmin(np.abs(a_grid[:, 0] - a))
            data_j = np.argmin(np.abs(r_grid[0, :] - r))
            val = data[data_i, data_j]

            if r <= 1:
                new[i, j] = val

    return new




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
        
        

"""     
        new = polar_to_cartesian(int1)
        fig, ax = plt.subplots()
        ax.imshow(new, origin='lower')
        
       
new = polar_to_cartesian(data)
fig, ax = plt.subplots()
ax.imshow(new, origin='lower')
"""
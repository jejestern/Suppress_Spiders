#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:13:49 2021

@author: jeje
"""


import cv2
import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt


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


        #--- ensure image is of the type float ---
        img = int1.astype(np.float32)

        #--- the following holds the square root of the sum of squares of the image dimensions ---
        #--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
        value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
        
  
        polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), 300, cv2.INTER_CUBIC)

        polar_image = polar_image.astype(np.uint8)
        
        # Choose the radius range
        polar_image = polar_image.T
        polar_img = polar_image[150:300, 0:]
        
        # Vertical flip image data to have the same convention as ds9
        axis2fl=int(polar_img.ndim-2)
        #print('axis to flip:',axis2fl)
        polar_img = np.flip(polar_img, axis2fl)
        
        fig, ax = plt.subplots()
        
        ax.imshow(polar_img, aspect='auto')
        plt.tight_layout()
        plt.show()

"""
        cv2.imshow("Polar Image", polar_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""
"""
Different methods to transform an image into polar coordinates.

Created on 2021-10-18
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits


def transform_to_polar(image):
    
    # Define the center of the image (position of the star)
    x_0, y_0 = image.shape
    x_0 = x_0/2
    y_0 = y_0/2
    
    r = np.sqrt(image)
    
    # Polar = [[], [], ...] where i = radius and j = theta
    polar = np.zeros_like(image)
    
    print(r)
    return 0


    
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
        
        transform_to_polar(int1)
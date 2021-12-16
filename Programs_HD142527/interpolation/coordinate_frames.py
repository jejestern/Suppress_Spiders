"""
09.12.2021

author: Jennifer Studer
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.visualization import simple_norm


# The image path of the images taken in the P2 mode
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

# Initialisation of lists in which we save the information
aperture_rad = 6.0

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
        
        # Define the norm for afterwards plotting the image
        norm = simple_norm(int1, 'log', percent=99.9)
        
        # Plot figure
        plt.figure()
        plt.imshow(int1, origin='lower', cmap='gray', norm = norm)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("Star_Img.pdf")
        plt.show()

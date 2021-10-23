"""
Different methods to transform an image into polar coordinates.
Atention: the x-axis is the vertical axis and the y-axis is the horizontal axis
in this case.

Created on 2021-10-18
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def cartesian_to_polar(x, y):
    start1 = timer()

    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(-y, x)
    
    # Needed so that phi = [0, 2*pi] otherwise phi = [-pi, pi]  
    phi %= (2*np.pi)
    
    end1 = timer()
    print("CartesiontoPolar :", end1-start1)
    return r, phi

def polar_corrdinates_grid(im_shape, center):
    
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    cx, cy = center[0], center[1] 

    # convert cartesian --> polar coordinates
    r_array, phi_array = cartesian_to_polar(x-cx, y-cy) 
    
    return r_array, phi_array


def radius_mask(r_array, radi):
    """
    Return a mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order and in radians.
    """

    # mask
    mask = (r_array <= radi[1]) & (r_array >= radi[0]) 

    return mask

def angle_mask(phi_array, angles):
    """
    Return a mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order and in radians.
    """

    # mask
    mask = (phi_array <= angles[1]) & (phi_array >= angles[0])

    return mask


def transform_to_polar(image, R_start, R_end):
    
    # Define the shape and of the image (position of the star)
    x_len, y_len = image.shape
    x_center = x_len/2 - 1
    y_center = y_len/2 - 1
    
    # Define the shape of the new coordinate system
    polar_len = R_end - R_start
    
    # Define the corresponding polar coordinates to the x-y coordinates
    r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))

    # Polar = [[], [], ...] where x-axis becomes phi and y-axis becomes radius
    polar = np.zeros((polar_len, polar_len)) * np.nan

    rad = np.linspace(R_start, R_end, polar_len+1)
    phi = np.linspace(0, 2*np.pi, polar_len+1)

    # We loop over the radi
    for j in range(polar_len):

        mask_r = radius_mask(r_array, (rad[j], rad[j+1]))

        for k in range(polar_len):
            
            mask_phi = angle_mask(phi_array, (phi[k], phi[k+1]))
            mask = mask_r & mask_phi
            
            polar[j][k] = sum(sum(image*mask))/len(np.where(mask == 1)[0])
        
        print(rad[j])

    return polar, rad, phi

    
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
        
        start5 = timer()
        img_polar, rads, phis = transform_to_polar(int1, 150, 300)
        end5 = timer()
        print("Transform to polar:", end5-start5)
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        
        mask_r = radius_mask(r_array, (150, 300))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=20)
        plt.colorbar()
        plt.show()

        plt.imshow(img_polar, origin='lower', cmap='gray')
        plt.colorbar()
        plt.show()
        

        file1 = open("radtophi_img.txt", "w") 
        for row in img_polar:
            np.savetxt(file1, row) 
        file1.close()
        
        file_rad = open("radi.txt", "w") 
        np.savetxt(file_rad, rads) 
        file_rad.close()

        file_phi = open("phis.txt", "w")             
        np.savetxt(file_phi, phis) 
        file_phi.close()
        

        
        #original_array = np.loadtxt("radtophi_img.txt").reshape(10, 10)


"""
Different functions needed to transform an image into polar coordinates.

Created on 2021-10-18
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
#from sys import argv, exit
#import os
#from astropy.io import fits
#import matplotlib.pyplot as plt
#from timeit import default_timer as timer
import scipy.ndimage 

def xy_to_rphi(x, y):
    """
    Coordinate transformation from cartesian to polar
    
    Parameters
    ----------
    x, y : float
        Cartesian coordinates

    Returns
    -------
    r, phi : float
        Polar coordinates, where phi (radians) is zero if (x=positiv, y=0) and 
        counterclock.
    """

    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(-x, y)
    
    # Needed so that phi = [0, 2*pi] otherwise phi = [-pi, pi]  
    phi %= (2*np.pi)
    
    return r, phi

def rphi_to_xy(r, phi):
    """
    Coordinate transformation from polar to cartesian.

    Parameters
    ----------
    r, phi : float
        Polar coordinates, where phi (radians) is zero if (x=positiv, y=0) and 
        counterclock.

    Returns
    -------
    x, y : float
        Cartesian coordinates.

    """
    
    x = -r * np.sin(phi)
    y = r * np.cos(phi)
    
    return x, y

def to_rphi_plane(f, im_shape, r_min, r_max):
    """
    Warping to r-phi plane.

    Parameters
    ----------
    f : float32, np.array 
        An intensity image.
    im_shape : (int, int)
        Shape of image f.
    r_min : int
        Inner radius.
    r_max : int
        Outer radius.

    Returns
    -------
    g : float32, np.array 
        r-phi plane image. 

    """
    
    r_len = r_max - r_min
    phi_len = int(2*np.pi*(r_min + r_len/2))
    #print(phi_len)
    #r_len = 1000
    #phi_len = 1000
    
    rs, phis = np.meshgrid(np.linspace(r_min, r_max, r_len), 
                           np.linspace(0, 2*np.pi, phi_len), sparse=True)
    
    xs, ys = rphi_to_xy(rs, phis)
    xs, ys = xs + im_shape[0]/2 -  1, ys + im_shape[1]/2 - 1
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    coords = np.vstack((ys, xs))
    
    g = scipy.ndimage.map_coordinates(f, coords, order=3)
    g = g.reshape(phi_len, r_len)
    
    return g

def from_rphi_plane(warped, im_shape, rmin, rmax):
    """
    Warping back from r-phi plane to cartesian coordinates

    Parameters
    ----------
    warped : float32, np.array 
        r-phi plane image. 
    im_shape : (int, int)
        Shape of the image before warping (shape of the output image).
    r_min : int
        Inner radius.
    r_max : int
        Outer radius.

    Returns
    -------
    h : float32, np.array 
        Cartesian image. 

    """
    
    phi_len, r_len = warped.shape
    
    xs, ys = np.meshgrid(np.arange(im_shape[1]), np.arange(im_shape[0]), sparse=True)
    
    rs, phis = xy_to_rphi(xs - (im_shape[0]/2 - 1), ys - (im_shape[1]/2 - 1))
    rs, phis = rs.reshape(-1), phis.reshape(-1)
    
    iis= phis / (2*np.pi) * (phi_len - 1)
    jjs= (rs - rmin) / (np.sqrt(r_len**2 + phi_len**2)) * (phi_len - 1)
    
    coords = np.vstack((iis, jjs))
    h = scipy.ndimage.map_coordinates(warped, coords, order=3)
    h = h.reshape(im_shape[0], im_shape[1])
    
    return h

def to_r_phi_plane(f, m, n, rmax, phimax):
    rs, phis = np.meshgrid(np.linspace(0, rmax,n), np.linspace(0, phimax, m),sparse=True)
    xs, ys = rphi_to_xy(rs, phis)
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    coords = np.vstack((ys, xs))
    print(coords)
    g = scipy.ndimage.map_coordinates(f, coords, order=3)
    g = g.reshape(m, n)
    return np.flipud(g)

def polar_corrdinates_grid(im_shape, center):
    
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    cx, cy = center[0], center[1] 

    # convert cartesian --> polar coordinates
    r_array, phi_array = xy_to_rphi(x-cx, y-cy) 
    
    return r_array, phi_array


def radius_mask(r_array, radi):
    """
    Return a radius mask.
    """

    # mask
    mask = (r_array <= radi[1]) & (r_array >= radi[0]) 

    return mask

def angle_mask(phi_array, angles):
    """
    Return a angle mask.
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
    R_len = R_end - R_start
    phi_len = int(R_start * np.pi)
    
    # Define the corresponding polar coordinates to the x-y coordinates
    r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))

    # Polar = [[], [], ...] where x-axis becomes phi and y-axis becomes radius
    polar = np.zeros((R_len, phi_len)) * np.nan

    rad = np.linspace(R_start, R_end, R_len+1)
    phi = np.linspace(0, 2*np.pi, phi_len+1)

    # We loop over the radi
    for j in range(R_len):

        mask_r = radius_mask(r_array, (rad[j], rad[j+1]))

        for k in range(phi_len):
            
            mask_phi = angle_mask(phi_array, (phi[k], phi[k+1]))
            mask = mask_r & mask_phi
            
            polar[j][k] = sum(sum(image*mask))/len(np.where(mask == 1)[0])
        
        print(rad[j])

    return polar, rad, phi

"""    
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
        
        start5 = timer()
        #img_polar, rads, phis = transform_to_polar(int1, R_1, R_2)
        end5 = timer()
        print("Transform to polar:", end5-start5)
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        
        mask_r = radius_mask(r_array, (R_1, R_2))
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
   

"""       
        #original_array = np.loadtxt("radtophi_img.txt").reshape(10, 10)


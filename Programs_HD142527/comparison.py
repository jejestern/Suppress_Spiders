#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the frequency spectrum of the various images. 

Created on 2022-02-01
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import aotools
from transformations_functions import polar_corrdinates_grid, radius_mask, angle_mask, rphi_to_xy
from aperture_fluxes import aperture_flux_image, aperture_flux_warped
from suppression_functions import suppress_division


# Choose the radial range into which we are warping the image
R_1 = 254
R_2 = 454

# We define the positions of the ghosts
gh_pos = [(891.0, 600.0), (213.0, 387.0)]

# Construct a PSF
zeros = np.zeros((1024, 1024))
mask_psf = aotools.circle(64,128)-aotools.circle(16, 128)
zeros[:128,:128] = mask_psf 
psf = aotools.ft2(zeros, delta=1./128.,)
psf = abs(psf)/1000

print(rphi_to_xy(354, np.pi/4))
psf_pos = (261, 761)
psf[psf_pos[1]-100:psf_pos[1]+100, psf_pos[0]-100:psf_pos[0]+100] = psf[512-100:512+100, 512-100:512+100]
psf[512-100:512+100, 512-100:512+100] = psf[0:200, 0:200]


## The object for which we will do the aperture flux calculations
model_planet = psf_pos #gh_pos[0] #

# This variable tells us, if we calculate the erro too, or not
error = False

########## The image path of the images taken in the P2 mode ############
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)


# Create a list in which we can save the information of the fft of the warped image
aper_origin = []
aper_warped = []
aper_flat = []
aper_cfreq = []
aper_lfreq = []

for image_name in files:
    if image_name.endswith("1.fits"): 
        # Reading in the images from camera 1
        img_data = fits.getdata(path + "/" + image_name, ext=0)
        fits.info(path + "/" + image_name)

        # Choose the intensity 1
        int1 = img_data[0,:,:]
        
        x_len, y_len = int1.shape
        x_center = x_len/2 - 1
        y_center = y_len/2 - 1
        
        # Add the PSF to the image
        int1 = int1 + psf
        
        # Choose the intensity
        Imax = 5
        Imax_small = 1
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        F_im, ap_im, annu_im = aperture_flux_image(int1, model_planet)
        print("The aperture flux of the model planet is: ", F_im)
        aper_origin.append(F_im)
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        ap_im.plot(color ='r', lw=1.0)
        annu_im.plot(color ='#0547f9', lw=1.0)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("comparison/HDimg_PSF.pdf")
        plt.show()
        
        # Spider suppression by using Gaussian division 
        shape, warped, flatten, flatten_c, flatten_l = suppress_division(int1, R_1, R_2)

        ## The different aspect ratio used for plotting
        aspect_rad = (2*np.pi/shape[0])/((R_2-R_1)/shape[1])

        ## Aperture flux of the model planet in the warped image
        F_w, _, _ = aperture_flux_warped(warped, shape, R_1, aspect_rad, model_planet)
        print("The aperture flux of the model planet in the warped image is: ", F_w)
        aper_warped.append(F_w)
        
        ## Aperture flux of the model planet in the flattened image
        F_f,  _, _ = aperture_flux_warped(flatten, shape, R_1, aspect_rad, model_planet)
        print("The aperture flux of the model planet in the flattened image is: ", F_f)
        aper_flat.append(F_f)

        ## Aperture flux of the model planet after radial central freq division 
        F_c, _, _ = aperture_flux_warped(flatten_c, shape, R_1, aspect_rad, model_planet)
        print("The aperture flux of the model planet with central radial freq division: ", F_c)
        aper_cfreq.append(F_c)

        ## Aperture flux of the model planet after suppression through divisioin
        F_l, _, _ = aperture_flux_warped(flatten_l, shape, R_1, aspect_rad, model_planet)
        print("The aperture flux of the model planet without spyders is: ", F_l)
        aper_lfreq.append(F_l)
        
        # Error Calculation with the help of Poisson distribution -> does not work...
        if error == True:
            
            image = np.where(int1.copy() <= 0, 0.1, int1.copy())
            
            for i in range(2):
                img = np.random.poisson(image)
                
                i += 1

# Save the aperture fluxes in a txt file
np.savez('comparison/aperture_fluxes.npz', name1=aper_origin, name2=aper_warped, 
         name3=aper_flat, name4=aper_cfreq, name5=aper_lfreq)
    
# Plot the outputs       
x_len = np.arange(len(aper_origin))

plt.figure()
plt.plot(x_len, aper_origin, 'x', label="Original Image")
plt.plot(x_len, aper_warped, 'x', label="Warped")
plt.plot(x_len, aper_flat, 'x', label="Flattened")
plt.plot(x_len, aper_cfreq, 'x', label="Suppressing central radial frequency")
plt.plot(x_len, aper_lfreq, 'x', label="Suppressing lower frequencies")
plt.xlabel("Images from HD142527")
plt.ylabel("Aperture flux of ghost 2")
plt.legend()
plt.tight_layout()
#plt.savefig("Ghost2_apertures.pdf")
plt.show()


ap_ratio = []
ap_ratio_f = []
for i in range(len(aper_origin)):
    ap_ratio.append(100/aper_origin[i]*aper_lfreq[i])
    ap_ratio_f.append(100/aper_flat[i]*aper_lfreq[i])
    
plt.figure()
plt.plot(x_len, ap_ratio, 'o', label="Original image to final one")
plt.plot(x_len, ap_ratio_f, 'o', label="Flattened image to final one")
plt.ylabel("Percentual aperture change")
plt.legend()
plt.show()

# We plot the percentual aperture change, starting by the flattened image
plt.figure()
plt.plot(x_len, 100/aper_flat*aper_flat, 'x', label="Flattened")
plt.plot(x_len, 100/aper_flat*aper_cfreq, 'x', label="Suppressing central radial frequency")
plt.plot(x_len, 100/aper_flat*aper_lfreq, 'x', label="Suppressing lower frequencies")
plt.xlabel("Images from HD142527")
plt.ylabel("Aperture change due to suppression: ghost 2  [%]")
plt.legend()
plt.tight_layout()
#plt.savefig("Ghost2_apertures_perc.pdf")
plt.legend()
plt.show()

        
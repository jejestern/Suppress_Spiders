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
from transformations_functions import polar_corrdinates_grid, to_rphi_plane, radius_mask, angle_mask
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from aperture_fluxes import aperture_flux_image, aperture_flux_warped
from filter_functions import e_func, gaussianBeam


# Create an image with only zeros with the same shape as the star images have
x_len, y_len = 1024, 1024
x_center = x_len/2 - 1
y_center = y_len/2 - 1

zeros = np.zeros((x_len, y_len))

# Choose the radial range into which we are warping the image
R_1 = 254
R_2 = 454
Imax = 1.0

# Warp the image into the r-phi plane
warp = to_rphi_plane(zeros, (x_len, y_len), R_1, R_2)
warp_or = warp.T
warp_shape = warp.shape

radi = np.arange(warp_shape[1])
phis = np.arange(warp_shape[0])/warp_shape[0]*2*np.pi
middle = int(R_1 + (R_2 - R_1)/2)


# Create the axis labels for the fft image -> physical frequencies
phi_freq = np.fft.fftfreq(warp_shape[0], d=2*np.pi/warp_shape[0])
phi_freq = np.fft.fftshift(phi_freq)
radi_freq = np.fft.fftfreq(warp_shape[1])
radi_freq = np.fft.fftshift(radi_freq)

# The different aspects used for plotting
aspect_value = (360/warp_shape[0])/((R_2-R_1)/warp_shape[1])
aspect_rad = (2*np.pi/warp_shape[0])/((R_2-R_1)/warp_shape[1])
aspect_freq = ((phi_freq[-1]-phi_freq[0])/warp_shape[0])/(
    (radi_freq[-1]-radi_freq[0])/warp_shape[1])
    

# We insert smoothed (gaussian) beams at the positions of the spyders
beamG1 = gaussianBeam(warp_or.copy(), 42, 11)
beamG2 = gaussianBeam(warp_or.copy(), 670, 16)
beamG3 = gaussianBeam(warp_or.copy(), 1155, 12)
beamG4 = gaussianBeam(warp_or.copy(), 1779, 12)

beamG = beamG1 + beamG2 + beamG3 + beamG4
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))
        



########## The image path of the images taken in the P2 mode ############
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

# We define the positions of the ghosts
gh_pos = [(891.0, 600.0), (213.0, 387.0)]

# Create a list in which we can save the information of the fft of the warped image
fouriers = []
aper_origin = []
aper_warped = []
aper_flat = []
aper_cfreq = []
aper_wspyd = []

for image_name in files[0:4]:
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
        
        # Choose the intensity
        Imax = 5
        Imax_small = 1
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        ## Computation of the aperture flux of the ghost
        model_planet = gh_pos[0] 
        f_ap_im, ap_im, annu_im = aperture_flux_image(int1, model_planet)
        print("The aperture flux of the model planet is: ", f_ap_im)
        aper_origin.append(f_ap_im)
        """
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        ap_im.plot(color ='r', lw=1.0)
        annu_im.plot(color ='#0547f9', lw=1.0)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
        """
        # Warp the image to the r-phi Plane    
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_or = warped.T
        warped_shape = warped.shape
        
        # Fourier transform the warped image
        fourier_w = np.fft.fftshift(np.fft.fft2(warped_or))
        
        ## Computation of the aperture flux of the model planet in the warped 
        ## image
        f_ap_w, ap_w_draw, annu_w_draw = aperture_flux_warped(warped_or, warped_shape, 
                                                              R_1, aspect_value, 
                                                              model_planet)
        print("The aperture flux of the model planet in the warped image is: ", f_ap_w)
        aper_warped.append(f_ap_w)
        

        # First we take out the intensity change in radial direction due to the
        # star in the center, by using an exponential fit function.
        
        ## Sum up the image along the phi axis
        r_trend = np.sum(warped_or, axis = 1)/warped_shape[0]

        ## Fitting an exponential and subtract it from the warped image
        popt, pcov = curve_fit(e_func, radi, r_trend)

        for i in range(warped_shape[0]):
            warped_or[:,i] = warped_or[:,i] - e_func(radi, *popt)
   
        ## Computation of the aperture flux of the model planet in the flattened 
        ## image
        f_ap_f, ap_f_draw, annu_f_draw = aperture_flux_warped(warped_or, warped_shape, 
                                                              R_1, aspect_value, 
                                                              model_planet)
        print("The aperture flux of the model planet in the flattened image is: ", f_ap_f)
        aper_flat.append(f_ap_f)
        
        ## Plot the output and its fft
        fourier_flat = np.fft.fftshift(np.fft.fft2(warped_or))
        """
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped_or, origin='lower', aspect=aspect_rad, vmin=0, 
                   vmax= Imax_small, extent=[0, 2*np.pi, R_1, R_2])
        ap_f_draw.plot(color ='r', lw=1.0)
        annu_f_draw.plot(color ='#0547f9', lw=1.0)
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier_flat), origin='lower', cmap='gray', 
                   norm=LogNorm(vmin=1), aspect=aspect_freq, 
                   extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
        
        plt.tight_layout()
        #plt.savefig("interpolation/HDflatten_R150_R300_4.pdf")
        plt.show()
        """
        ## Save the fourier transform in the list
        fouriers.append(abs(fourier_flat))
        
        # Subtract Gaussian 
        ### Take out some structure via fft: SUBTRACTION of gaussian of the 
        ### center frequencyies (radial)
        w = 59 # With this value we have the smallest aperture flux loss (ghost)
        q = 100
        spyd_center = fourier_flat[middle-R_1, :]
        spyd_center[int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_center[int(len(phis)/2)-
                                                                         w:int(len(phis)/2)
                                                                         +w]/fft_beamG[middle-R_1, int(len(phis)/2)-w:int(len(phis)/2)+w]

        spyd_center[:int(len(phis)/2)-2*w] = spyd_center[:int(len(phis)/2)-2*w]/q
        spyd_center[int(len(phis)/2)+2*w:] = spyd_center[int(len(phis)/2)+2*w:]/q
        
        """
        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spyd_center), label="ratio")
        #plt.xlim((-20, 20))
        #plt.ylim((10**(-1), 10**(2)))
        plt.title("FFT ratio of beam images")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend()
        plt.show()
        """
        
        fourier_flat[middle-R_1, :] = spyd_center
        
        fft_back_spyd_center = abs(np.fft.ifft2(fourier_flat))
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back where a gaussian is subtracted from the center
        f_ap_fft, ap_fft_draw, annu_fft_draw = aperture_flux_warped(fft_back_spyd_center, 
                                                                    warped_shape, R_1, 
                                                                    aspect_value, 
                                                                    model_planet)
        print("The aperture flux of the model planet without (central frequencies only) spyders is: ", f_ap_fft)
        aper_cfreq.append(f_ap_fft)

        
        # Ratio in radial direction in order to make a Gaussian subtraction of all 
        # frequencies in radial direction (also the larger ones)
        ratio_gaussian = sum(abs(fourier_flat[middle-R_1, int(len(phis)/2)-
                                              w:int(len(phis)/2)+w]))/sum(
                                                  abs(fourier_flat[20, int(len(phis)/2)-w:int(len(phis)/2)+w]))
        print("Ratio for the small gaussian: ", ratio_gaussian)
    
    
        small_gaussian = fft_beamG[middle-R_1, :] * ratio_gaussian
        """
        y = 0
        plt.figure(figsize=(8, 16*aspect_value))
        while y < (R_2-R_1)/2:
            plt.semilogy(phi_freq, abs(fourier_flat[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
            y += 20
        plt.semilogy(phi_freq, abs(small_gaussian + 0.0001), label="Gaussian beam")
        plt.ylim((10**(-1), 10**(4)))
        plt.title("FFT of beam images horizontal")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        """
        w = 10
        h = 5
        spyd_low = fourier_flat[middle-R_1-h:middle-R_1, :] 
        spyd_low[:, int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_low[:, int(len(phis)/2)
                                                                      -w:int(len(phis)/2)
                                                                      +w]/small_gaussian[int(len(phis)/2)-w:int(len(phis)/2)+w]
        spyd_high = fourier_flat[middle-R_1:middle-R_1+h, :] 
        spyd_high[:, int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_high[:, int(len(phis)/2)
                                                                        -w:int(len(phis)/2)
                                                                        +w]/small_gaussian[int(len(phis)/2)-w:int(len(phis)/2)+w]

        neg_r = 40
        neg_a = 300
        fourier_flat[:middle-R_1-neg_r, :] = fourier_flat[:middle-R_1-neg_r, :]/q
        fourier_flat[middle-R_1+neg_r:, :] = fourier_flat[middle-R_1+neg_r:, :]/q
        fourier_flat[:, :int(len(phis)/2)-neg_a] = fourier_flat[:,  :int(len(phis)/2)-neg_a]/q
        fourier_flat[:, int(len(phis)/2)+neg_a:] = fourier_flat[:, int(len(phis)/2)+neg_a:]/q
        #spyd_center[:int(len(phis)/2)-w] = spyd_center[:int(len(phis)/2)-w]/q
        #spyd_center[int(len(phis)/2)+w:] = spyd_center[int(len(phis)/2)+w:]/q
        """
        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spyd_low[h-1]), label="ratio")
        #plt.xlim((-20, 20))
        #plt.ylim((10**(-2), 10**(3)))
        plt.title("FFT ratio of beam images")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.semilogy(radi_freq, abs(fourier_flat[:, int(len(phis)/2)] + 0.0001), label="vertical cut")
        #plt.ylim((10**(-1), 10**(5)))
        plt.title("FFT of beam images vertical")
        plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
        plt.show()
        """

        fourier_flat[middle-R_1-h:middle-R_1, :] = spyd_low
        fourier_flat[middle-R_1:middle-R_1+h, :] = spyd_high
        fft_back_spyd = abs(np.fft.ifft2(fourier_flat))
        

        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back image where the spyders are taken away
        f_ap_fft, ap_fft_draw, annu_fft_draw = aperture_flux_warped(fft_back_spyd, 
                                                                    warped_shape, R_1, 
                                                                    aspect_value, 
                                                                    model_planet)
        print("The aperture flux of the model planet without spyders is: ", f_ap_fft)
        aper_wspyd.append(f_ap_fft)

        # Plotting the back transformation
        plt.figure(figsize=(8, 16*aspect_value))
        
        plt.subplot(211)
        plt.imshow(abs(fourier_flat + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
                   aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()

        
        plt.subplot(212)
        plt.imshow(fft_back_spyd, origin='lower', aspect=aspect_rad, vmin=0, vmax=0.5, 
                   extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                          r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        

      
plt.figure(figsize=(8, 16*aspect_value))
for fft in fouriers:
    plt.semilogy(phi_freq, fft[middle-R_1, :])
#plt.xlim(-50, 50)
plt.title("FFT for radial frequency 0 for different images")  
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.show()    

plt.figure(figsize=(8, 16*aspect_value))
plt.plot(phi_freq, fouriers[0][middle-R_1, :]-fouriers[1][middle-R_1, :])
#plt.xlim(-50, 50)
plt.title("FFT for radial frequency 0 difference of different images")  
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.show() 
        
plt.figure()
for fft in fouriers: 
    plt.semilogy(radi_freq, fft[:, int(len(phis)/2)])
plt.title("FFT for angular frequency 0 for different images")
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.show()

plt.figure()
plt.plot(radi_freq, fouriers[0][:, int(len(phis)/2)]-fouriers[1][:, int(len(phis)/2)])
plt.title("FFT for angular frequency 0 difference of different images")
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.show()

x_len = np.arange(len(aper_origin))

plt.figure()
plt.plot(x_len, aper_origin, 'o', label="aperture flux image")
plt.plot(x_len, aper_warped, 'o', label="aperture flux warped")
plt.plot(x_len, aper_flat, 'o', label="aperture flux flattend")
plt.plot(x_len, aper_cfreq, 'o', label="aperture flux witout central freq")
plt.plot(x_len, aper_wspyd, 'o', label="aperture flux spyders filtered out")
plt.legend()
plt.show()


ap_ratio = []
for i in range(len(aper_origin)):
    ap_ratio.append(100 - 100/aper_origin[i]*aper_wspyd[i])
    
plt.figure()
plt.plot(x_len, ap_ratio, 'o', label="percentual aperture loss")
plt.legend()
plt.show()


"""        
        ### Plot vertical and horizontal cuts throught the image and the FFT
        y = 0
        plt.figure(figsize=(8, 16*aspect_value))
        while y < (R_2-R_1)/2:
            plt.semilogy(phi_freq, abs(fourier_flat[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
            y += 20
        plt.semilogy(phi_freq, abs(fourier_flat[middle-R_1, :] + 0.0001), label="radial freq. = 0")
        plt.ylim((10**(-1), 10**(5)))
        plt.title("FFT of beam images horizontal")  
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        x = 0
        plt.figure()
        while x < len(phis)/2:
            plt.semilogy(radi_freq, abs(fourier_flat[:, x] + 0.0001), label ="phi pos. = %.1f" %(phi_freq[x]))
            x += 240
        plt.semilogy(radi_freq, abs(fourier_flat[:, int(len(phis)/2)] + 0.0001), label="phi pos. = 0")
        plt.ylim((10**(-1), 10**(5)))
        plt.title("FFT of beam images horizontal")
        plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
"""
        
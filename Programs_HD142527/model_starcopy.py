#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Include a model planet (copy of the star, just a lot fainter) and investigate
the effect on the FFT (as well as on the warping).

Created on 2021-12-20
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
from sys import argv, exit
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from transformations_functions import polar_corrdinates_grid, to_rphi_plane, radius_mask, angle_mask, from_rphi_plane
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
"""        
# Plot the created image  
plt.imshow(zeros, origin='lower', cmap='gray', vmin=0, vmax=Imax)
plt.colorbar()
plt.tight_layout()
plt.show()
"""
# Warp the image into the r-phi plane
warp = to_rphi_plane(zeros, (x_len, y_len), R_1, R_2)
warp_or = warp.T
warp_shape = warp.shape

aspect_value = (360/warp_shape[0])/((R_2-R_1)/warp_shape[1])
aspect_rad = (2*np.pi/warp_shape[0])/((R_2-R_1)/warp_shape[1])


radi = np.arange(warp_shape[1])
phis = np.arange(warp_shape[0])/warp_shape[0]*2*np.pi
middle = int(R_1 + (R_2 - R_1)/2)

# We insert beams at the positions of the spyders
beams = warp_or.copy()
beams[:, 31:53] = 1
beams[:, 654:686] = 1
beams[:, 1143:1167] = 1
beams[:, 1767:1791] = 1
        
# Fourier transform the warped image with beams
fft_beams = np.fft.fftshift(np.fft.fft2(beams))

# Create the axis labels for the fft image
phi_freq = np.fft.fftfreq(warp_shape[0], d=2*np.pi/warp_shape[0])
phi_freq = np.fft.fftshift(phi_freq)
radi_freq = np.fft.fftfreq(warp_shape[1])
radi_freq = np.fft.fftshift(radi_freq)

aspect_freq = ((phi_freq[-1]-phi_freq[0])/warp_shape[0])/(
    (radi_freq[-1]-radi_freq[0])/warp_shape[1])
    
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))

plt.subplot(211)
plt.imshow(beams, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
#plt.imshow(abs(fft_beams + 0.0001), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
 #          aspect=aspect_value, extent=[0, 360, R_1, R_2])
plt.imshow(abs(fft_beams + 0.0001), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
#plt.savefig("fourier/HDwarped_R254_R454.pdf")
plt.show()

# We insert smoothed (gaussian) beams at the positions of the spyders
beamG1 = gaussianBeam(warp_or.copy(), 42, 11)
beamG2 = gaussianBeam(warp_or.copy(), 670, 16)
beamG3 = gaussianBeam(warp_or.copy(), 1155, 12)
beamG4 = gaussianBeam(warp_or.copy(), 1779, 12)

beamG = beamG1 + beamG2 + beamG3 + beamG4
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(beamG, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_beamG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()








# This part takes the argument 
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
        
        # Choose the intensity
        Imax = 5
        Imax_small = 1
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
              
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_or = warped.T
        warped_shape = warped.shape
        
        # Fourier transform the warped image
        fourier_w = np.fft.fftshift(np.fft.fft2(warped_or))
        
        # Plotting the warped and fft of it
        plt.figure(figsize=(8, 16*aspect_value))
        
        plt.subplot(211)
        plt.imshow(warped_or, origin='lower', aspect=aspect_rad, vmin=0, 
                       vmax=Imax, extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier_w), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
                   aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], 
                                               radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
        
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        
        
        # We cut out the star and insert it a lot less bright in the top left 
        # part of the image
        intens = 0 #10**(-3)
        int1[550:950, 50:450] += int1[int(y_center-200):int(y_center+200), 
                                      int(x_center-200):int(x_center+200)]*intens 
        
        ## Computation of the aperture flux of the model planet
        model_planet = gh_pos[0] #[250, 
        f_ap_im, ap_im, annu_im = aperture_flux_image(int1, model_planet)
        print("The aperture flux of the model planet is: ", f_ap_im)
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        ap_im.plot(color ='r', lw=1.0)
        annu_im.plot(color ='#0547f9', lw=1.0)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
              
        warped_model = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_m_or = warped_model.T
        
        ## Computation of the aperture flux of the model planet in the warped 
        ## image
        f_ap_w, ap_w_draw, annu_w_draw = aperture_flux_warped(warped_m_or, warped_shape, 
                                                              R_1, aspect_rad, 
                                                              model_planet)
        print("The aperture flux of the model planet in the warped image is: ", f_ap_w)
        
        ## Fourier transform the warped image (model star)
        fourier_m_w = np.fft.fftshift(np.fft.fft2(warped_m_or))
        
        ## Plotting the warped image and its fft (model planet)
        plt.figure(figsize=(8, 16*aspect_value))
        
        plt.subplot(211)
        plt.imshow(warped_m_or, origin='lower', aspect=aspect_rad, vmin=0, 
                       vmax=Imax, extent=[0, 2*np.pi, R_1, R_2])
        ap_w_draw.plot(color ='r', lw=1.0)
        annu_w_draw.plot(color ='#0547f9', lw=1.0)
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier_m_w), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
                   aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], 
                                               radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
       
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R150_R300_15.pdf")
        plt.show()
        """
        # Plot and calculate the fft(model_star)-fft 
        fft_model = fourier_m_w - fourier_w
        plt.figure(figsize=(8, 8*aspect_value))
        plt.imshow(abs(fft_model), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
                   aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], 
                                               radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        """
        """ 
        We now want to apply different methods to the images (with and without 
        model planet) to see what happens and if the stars flux apperture is 
        conserved.
        """
        
        # First we take out the intensity change in radial direction due to the
        # star in the center, by using an exponential fit function.
        
        ## Sum up the image along the phi axis
        r_trend = np.sum(warped_m_or, axis = 1)/warped_shape[0]

        ## Fitting an exponential
        popt, pcov = curve_fit(e_func, radi, r_trend)

        ## Plot the fit 
        plt.figure()
        plt.plot(radi, r_trend, 'b.', 
                 label="mean angular intensity distribution")
        plt.plot(radi, e_func(radi, *popt), 'r-', label='exponential fit') #': a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.xlabel(r'Radius')
        plt.ylabel('Mean angular intensity')
        plt.legend()
        #plt.savefig("interpolation/mean_angular_intensity_R150_300.pdf")
        plt.show()

        for i in range(warped_shape[0]):
            warped_m_or[:,i] = warped_m_or[:,i] - e_func(radi, *popt)
   
        ## Computation of the aperture flux of the model planet in the flattened 
        ## image
        f_ap_f, ap_f_draw, annu_f_draw = aperture_flux_warped(warped_m_or, warped_shape, 
                                                              R_1, aspect_rad, 
                                                              model_planet)
        print("The aperture flux of the model planet in the flattened image is: ", f_ap_f)
        
        ## Plot the output
        fourier_flat = np.fft.fftshift(np.fft.fft2(warped_m_or))
        
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped_m_or, origin='lower', aspect=aspect_rad, vmin=0, 
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
        
        ## Investigate the shape and intensity of the spikes
        ## Plot the 
        plt.figure()
        plt.plot(radi_freq, warped_m_or[:, 42], '.', label="intensity distribution x=42")
        plt.plot(radi_freq, warped_m_or[:, 670], '.', label="intensity distribution x=670")
        plt.plot(radi_freq, warped_m_or[:, 1155], '.', label="intensity distribution x=1155")
        plt.plot(radi_freq, warped_m_or[:, 1779], '.', label="intensity distribution x=1779")
        plt.legend()
        plt.show()
        
        ### Plot vertical and horizontal cuts throught the image and the FFT
        y = 0            
        plt.figure(figsize=(8, 16*aspect_value))
        while y < (R_2-R_1): 
            plt.plot(phis, warped_m_or[y, :], label="at y = %.0f" %(y+R_1))
            y += 50
        plt.title("Horizontal cut through the beam images")
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                          r'$3\pi/2$', r'$2\pi$'])
        plt.legend()
        plt.show()

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
        
        
        ### Take out some structure via fft: SUBTRACTION of gaussian of the 
        ### center frequencyies (radial)
        w = 60
        q = 10
        spyd_center = fourier_flat[middle-R_1, :]
        spyd_center[int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_center[int(len(phis)/2)-
                                                                         w:int(len(phis)/2)
                                                                         +w]/fft_beamG[middle-R_1, int(len(phis)/2)-w:int(len(phis)/2)+w]

        spyd_center[:int(len(phis)/2)-w] = spyd_center[:int(len(phis)/2)-w]/q
        spyd_center[int(len(phis)/2)+w:] = spyd_center[int(len(phis)/2)+w:]/q

        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spyd_center), label="ratio")
        #plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
        #plt.xlim((-20, 20))
        plt.ylim((10**(-1), 10**(2)))
        plt.title("FFT ratio of beam images")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend()
        plt.show()
        
        fourier_flat[middle-R_1, :] = spyd_center
        fft_back_spyd_center = abs(np.fft.ifft2(fourier_flat))
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back where a gaussian is subtracted from the center
        f_ap_fft, ap_fft_draw, annu_fft_draw = aperture_flux_warped(fft_back_spyd_center, 
                                                                    warped_shape, R_1, 
                                                                    aspect_value, 
                                                                    model_planet)
        print("The aperture flux of the model planet without spyders is: ", f_ap_fft)

        # Plotting the back transformation
        plt.figure(figsize=(8, 16*aspect_value))
        
        plt.subplot(211)
        plt.imshow(abs(fourier_flat + 0.0001), origin='lower', cmap='gray',  
                   norm=LogNorm(vmin=1), aspect=aspect_freq, 
                   extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()

        
        plt.subplot(212)
        plt.imshow(fft_back_spyd_center, origin='lower', aspect=aspect_rad, 
                   vmin=0, vmax=0.5, extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                          r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()

        plt.tight_layout()
        #plt.savefig("interpolation/HDwarped_R290_R490.pdf")
        plt.show()
        
        
        
        
        # Ratio in radial direction in order to make a Gaussian subtraction of all 
        # frequencies in radial direction (also the larger ones)
        ratio_gaussian = sum(abs(fourier_flat[middle-R_1, int(len(phis)/2)-
                                              w:int(len(phis)/2)+w]))/sum(
                                                  abs(fourier_flat[20, int(len(phis)/2)-w:int(len(phis)/2)+w]))
        print(ratio_gaussian)
    
    
        small_gaussian = fft_beamG[middle-R_1, :] * ratio_gaussian

        y = 0
        plt.figure(figsize=(8, 16*aspect_value))
        while y < (R_2-R_1)/2:
            plt.semilogy(phi_freq, abs(fourier_flat[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
            y += 20
        #plt.semilogy(phi_freq, abs(fft_spydG[middle-R_1, :] + 0.0001), label="radial freq. = 0")
        #plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beam")
        plt.semilogy(phi_freq, abs(small_gaussian + 0.0001), label="Gaussian beam")
        plt.ylim((10**(-1), 10**(4)))
        plt.title("FFT of beam images horizontal")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        w = int(w/2)
        h = 3
        spyd_low = fourier_flat[middle-R_1-h:middle-R_1, :] 
        spyd_low[:, int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_low[:, int(len(phis)/2)
                                                                      -w:int(len(phis)/2)
                                                                      +w]/small_gaussian[int(len(phis)/2)-w:int(len(phis)/2)+w]
        spyd_high = fourier_flat[middle-R_1:middle-R_1+h, :] 
        spyd_high[:, int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_high[:, int(len(phis)/2)
                                                                        -w:int(len(phis)/2)
                                                                        +w]/small_gaussian[int(len(phis)/2)-w:int(len(phis)/2)+w]

        #spyd_center[:int(len(phis)/2)-w] = spyd_center[:int(len(phis)/2)-w]/q
        #spyd_center[int(len(phis)/2)+w:] = spyd_center[int(len(phis)/2)+w:]/q

        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spyd_low[h-1]), label="ratio")
        #plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
        #plt.xlim((-20, 20))
        #plt.ylim((10**(-2), 10**(3)))
        plt.title("FFT ratio of beam images")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend()
        plt.show()

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

        
"""          

        h2 = from_rphi_plane(warped, (x_len, y_len), R_1, R_2)
        plt.imshow(h2, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        plt.colorbar()
        plt.show()
        
        
        plt.imshow(h2-int1*mask, origin='lower', cmap='gray', vmin=-0.1, vmax=0.1)
        plt.colorbar()
        plt.show()  
"""
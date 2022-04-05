#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-03-26
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from transformations_functions import rphi_to_xy, polar_corrdinates_grid, to_rphi_plane, radius_mask, angle_mask
from scipy.optimize import curve_fit
from aperture_fluxes import aperture_flux_image, aperture_flux_warped
from filter_functions import e_func, Gaussian1D
import aotools
from suppression_functions import fourier_plotting, suppress_subtraction
from scipy.signal import find_peaks


# Choose the radial range into which we are warping the image
R_1 = 254
R_2 = 454

# Choose the intensity
Imax = 5
Imax_small = 0.5
Imin_small = -0.5
        


########## The image path of the images taken in the P2 mode ##################
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

# We define the positions of the ghosts
gh_pos = [(891.0, 599.0), (213.0, 387.0)]


for image_name in files[0:3]:
    if image_name.endswith("1.fits"): 
        # Reading in the images from camera 1
        img_data = fits.getdata(path + "/" + image_name, ext=0)
        fits.info(path + "/" + image_name)

        # Choose the intensity 1
        int1 = img_data[0,:,:]
        
        x_len, y_len = int1.shape
        x_center = x_len/2 - 1
        y_center = y_len/2 - 1
    
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        # For checking we can insert a PSF
        zeros = np.zeros((x_len, y_len))
        mask_psf = aotools.circle(64,128)-aotools.circle(16, 128)
        zeros[:128,:128] = mask_psf 
        psf = aotools.ft2(zeros, delta=1./128.,)
        psf = abs(psf)/10
        
        psf_pos = rphi_to_xy(354, 1.95) 
        print(psf_pos)
        psf_pos = (int(psf_pos[0]+512), int(psf_pos[1]+512)) #(300, 800)
        print(psf_pos)
        psf[psf_pos[1]-100:psf_pos[1]+100, psf_pos[0]-100:psf_pos[0]+100] = psf[512-100:512+100, 512-100:512+100]
        psf[512-100:512+100, 512-100:512+100] = psf[0:200, 0:200]
        
        #int1 = int1 + psf
        
        ## Computation of the aperture flux of the ghost/PSF
        model_planet = psf_pos #gh_pos[0] #
        f_ap_im, ap_im, annu_im = aperture_flux_image(int1, model_planet)
        print("The aperture flux of the model planet in the original image is: ", f_ap_im)
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        ap_im.plot(color ='r', lw=1.0)
        annu_im.plot(color ='#0547f9', lw=1.0)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
        
        #################### Warping ########################################
        # Warp the image to the r-phi Plane    
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_shape = warped.shape
        warped = warped.T
        
        # Define the radial and angular coordinates
        radi = np.arange(warped_shape[1])
        phis = np.arange(warped_shape[0])/warped_shape[0]*2*np.pi
        cen_r = int((R_2-R_1)/2)
        cen_phi = int(warped_shape[0]/2)
            
        # Create the axis labels for the fft image -> physical frequencies
        phi_freq = np.fft.fftfreq(warped_shape[0], d=2*np.pi/warped_shape[0])
        phi_freq = np.fft.fftshift(phi_freq)
        radi_freq = np.fft.fftfreq(warped_shape[1])
        radi_freq = np.fft.fftshift(radi_freq)
        
        # The different aspect ratios used for plotting
        aspect_value = (360/warped_shape[0])/((R_2-R_1)/warped_shape[1])
        aspect_rad = (2*np.pi/warped_shape[0])/((R_2-R_1)/warped_shape[1])
        aspect_freq = ((phi_freq[-1]-phi_freq[0])/warped_shape[0])/(
            (radi_freq[-1]-radi_freq[0])/warped_shape[1])

        
        ## Computation of the aperture flux of the model planet in the warped 
        ## image
        f_ap_w, ap_w_draw, annu_w_draw = aperture_flux_warped(warped, warped_shape, 
                                                              R_1, aspect_value, 
                                                              model_planet)
        print("The aperture flux of the model planet in the warped image is: ", f_ap_w)
        
        
        ########################## Flattening ################################
        # First we take out the intensity change in radial direction due to the
        # star in the center, by using an exponential fit function.
        
        ## Sum up the image along the phi axis
        r_trend = np.sum(warped, axis = 1)/warped_shape[0]

        ## Fitting an exponential and subtract it from the warped image
        popt, pcov = curve_fit(e_func, radi, r_trend)

        for i in range(warped_shape[0]):
            warped[:,i] = warped[:,i] - e_func(radi, *popt)
   
        ## Computation of the aperture flux of the model planet in the flattened 
        ## image
        f_ap_f, ap_f_draw, annu_f_draw = aperture_flux_warped(warped, warped_shape, 
                                                              R_1, aspect_rad, 
                                                              model_planet)
        print("The aperture flux of the model planet in the flattened image is: ", f_ap_f)
        print("This corresponds to 100 %")

        ## Plot the output and its fft
        fourier = np.fft.fftshift(np.fft.fft2(warped))
        fourier_plotting(warped, fourier, R_1, R_2, phi_freq, radi_freq, Imin_small, 
                         Imax_small, fourier_enl=[(-20, 20), (-0.06, 0.06)])
        
        """
        ## Plot the frequency ranges
        y = int((R_2-R_1)/2)
        plt.figure(figsize=(8, 24*aspect_value))
        while y > 0:
            plt.semilogy(phi_freq, abs(fourier[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
            if y == int((R_2-R_1)/2):
                y -= 1
            elif abs(radi_freq[y]) < 0.02:
                y -= 5
            else:
                y -= 40
        #plt.xlim((-30, 30))
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        #plt.savefig("suppression/rad0.pdf")
        plt.show()
        
        x = int(len(phis)/2)
        plt.figure(figsize=(8, 5))
        while x > 0:
            plt.semilogy(radi_freq, abs(fourier[:, x]), label ="phi pos. = %.i" %(phi_freq[x]))
            if abs(phi_freq[x]) > 20:
                x -= 500
            else:
                x -= 60
        #plt.ylim((10**(-1), 10**(5)))
        plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        #plt.savefig("suppression/ang0.pdf")
        plt.show()
        """
        ######################################################################
        ########################### Subtraction ##############################
        ######################################################################
        ## Plot the total intensity of the image along the angular axis
        phi_trend = np.sum(warped, axis = 0)/warped_shape[0]
        
        plt.figure()
        plt.plot(phis, phi_trend)
        plt.show()
        
        # The position of the spider
        spos = find_peaks(phi_trend, distance=480)[0]
        #degsym = int(180/360*warped_shape[0]) # 180 degree symmetry
        #degsep = 485 # 
        #spos = [42,  670, 42+degsym, 670+degsym]
        print(spos)
        
        # We find the width and the maximal intensity of the spiders around the 
        # central radius
        thick_cen = 10
        warped_center = warped[cen_r -thick_cen:cen_r+thick_cen, :]/(2*thick_cen)
        phi_center = np.sum(warped_center, axis = 0)
        
        Intens = []
        I_2 = []
        for i in range(4):
            n = 4 
            i_2 = np.max(phi_center[spos[i]-n:spos[i]+n])
            I_2.append(i_2)  
            
            intens = np.max(warped[cen_r, spos[i]-n:spos[i]+n])
            Intens.append(intens)  
        

        print(Intens)
        print(I_2)
        
        # We insert smoothed gaussian (1D) at the positions of the spiders with 
        # the same width and intensity
        width_hand = [5, 10, 8, 6]  # Guessed by hand
        width = []
        print(width_hand[0]/warped_shape[0]*2*np.pi)
        g = 0
        for i in range(4):
            n = 15
            phis_n = phis[spos[i]-n:spos[i]+n].copy()
            phi_center_n = phi_center[spos[i]-n:spos[i]+n].copy()

            parameters, covariance = curve_fit(Gaussian1D, phis_n.copy(), phi_center_n.copy())
            print(parameters)
            width.append(parameters[1])
            fit_y = Gaussian1D(phis.copy(), parameters[0]+spos[i]-n, parameters[1], parameters[2])
            """
            plt.figure()
            plt.plot(phis_n, phi_center_n, '.')
            plt.plot(phis_n, fit_y[spos[i]-n:spos[i]+n])
            plt.show()
            """
            #g_i = Gaussian1D(phis.copy(), spos[i], width_hand[i], Intens[i])  # BY eye
            #g += g_i
            g += fit_y
            
        plt.figure()
        plt.plot(phis/(2*np.pi)*warped_shape[0], phi_center)
        plt.plot(phis/(2*np.pi)*warped_shape[0], g)
        #plt.xlim((0.0, 0.3))
        plt.show()
            
        fft_g = np.fft.fftshift(np.fft.fft(g))
        
        # We want to plot  it
        fig1, ax1 = plt.subplots(2, 1, figsize=(8, 8))  
        ax1[0].plot(phis, g, 
                    label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
                    %(5/warped_shape[0]*2*np.pi, 10/warped_shape[0]*2*np.pi, 8/warped_shape[0]*2*np.pi, 
                      6/warped_shape[0]*2*np.pi))
        ax1[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                                 r'$3\pi/2$', r'$2\pi$'])
        ax1[0].set_xlabel(r'$\varphi$ [rad]')
        ax1[0].legend(loc='upper right')
        
        ax1[1].plot(phi_freq, fft_g.real, label="FFT")
        ax1[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        #ax1[1].set_ylim((10**(-3), 4*10**(4)))
        ax1[1].set_xlim((-20, 20))
        ax1[1].legend()
        #plt.savefig("fourier/Gaussian_fourdiffspyders.pdf")
        plt.show()
        
        # We want to know approx how much of the total intensity (=central 
        # frequency) is caused by the spiders
        I_tot = 0
        warped_I = warped.copy()
        for i in range(len(width)):
            I_tot += np.sum(warped[:, spos[i]-int(width[i]):spos[i]+int(width[i])])
            warped_I[:, spos[i]-int(width[i]):spos[i]+int(width[i])] = 10
            
        #fourier_I = np.fft.fftshift(np.fft.fft2(warped_I))
        #fourier_plotting(warped_I, fourier_I, R_1, R_2, phi_freq, radi_freq, Imin_small, 
         #                Imax_small, fourier_enl=[(-20, 20), (-0.06, 0.06)])
        
        # Factor which describes the intensity difference between 1D and reality
        fac = I_tot/fft_g[cen_phi]

        #Subtraction
        fourier_sub = fourier.copy()
        spid_center = fourier[cen_r, :].copy()
        spid_center = spid_center - fft_g*fac
        fourier_sub[cen_r, :] = spid_center
        
        ## Plot the frequency ranges
        plt.figure(figsize=(8, 4))
        plt.plot(phi_freq, fourier[cen_r, :].real, label ="radial freq. = %.2f" %(radi_freq[cen_r]))
        plt.plot(phi_freq, fourier_sub[cen_r, :].real, label ="Spider subtracted" %(radi_freq[cen_r]))
        plt.xlim((-20, 20))
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        #plt.savefig("suppression/rad0.pdf")
        plt.show()
        
        ## Plot the output and its fft
        warped_back = np.fft.ifft2(np.fft.ifftshift(fourier_sub)).real
        fourier_plotting(warped_back, fourier_sub, R_1, R_2, phi_freq, radi_freq, Imin_small, 
                         Imax_small, fourier_enl=[(-20, 20), (-0.06, 0.06)], 
                         savefig="subtraction/HDsubtracted.pdf")
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back where a gaussian is subtracted from the center
        f_ap_sub, _, _ = aperture_flux_warped(warped_back, warped_shape, R_1, 
                                              aspect_rad, model_planet)
        print("The aperture flux of the model planet without (only at radial freq=0) spyders is: ", f_ap_sub)
        print("This corresponds to ", round(100/f_ap_f*f_ap_sub, 3), " %")
        
        ######################################################################
        ######### Frequency suppression also in radial direction #############
        ######################################################################
        
        ## Plot the frequency ranges
        y = cen_r
        plt.figure(figsize=(8, 24*aspect_value))
        while y > cen_r -5:
            plt.plot(phi_freq, abs(fourier[y, :]), label ="radial freq. = %.3f" %(radi_freq[y]))
            y -= 1
        plt.xlim((-20, 20))
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        #plt.savefig("suppression/rad0.pdf")
        plt.show()
        
        # Ratio in radial direction in order to make a Gaussian subtraction of all 
        # frequencies in radial direction (also the larger ones)
        neg_r = 40
        w = 50
        ratio_gauss = np.sum(abs(fourier[cen_r+1:cen_r+neg_r, int(len(phis)/2)-w:
                                         int(len(phis)/2)-1]), axis=1)/np.sum(abs(fourier[cen_r,
                                             int(len(phis)/2)-w:int(len(phis)/2)-1]))        
        #print("Ratio for the small gaussian: ", ratio_gauss)
        ratio_spi = np.sum(abs(fourier[:, cen_phi-3:cen_phi+3]), axis=1)/np.sum(
            abs(fourier[cen_r, cen_phi-3:cen_phi+3]))

    
        y = cen_r - 1
        y_g = 0
        h = 3
        plt.figure(figsize=(8, 16*aspect_value))
        while y > cen_r - h:
            plt.plot(phi_freq, abs(fourier[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
            plt.plot(phi_freq, abs(fft_g*fac*ratio_spi[y]), label="Simulation")
            y -= 1
        plt.xlim((-20, 20))
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
    
        r_n = cen_r - 1
        r_p = cen_r + 1
        ratio_i = 0
        fourier_sup = fourier_sub.copy()
        while r_n > cen_r - h:
            
            spid = fourier[r_n, :].copy()
            spid = spid - fft_g*fac*ratio_gauss[ratio_i]
            fourier_sup[r_n, :] = spid
            
            spid = fourier[r_p, :].copy()
            spid = spid - fft_g*fac*ratio_gauss[ratio_i]
            fourier_sup[r_p, :] = spid
            
            r_n -= 1
            r_p += 1
            ratio_i += 1
            
        warped_sup = np.fft.ifft2(np.fft.ifftshift(fourier_sup)).real 
        
        ## Plot the output and its fft
        fourier_plotting(warped_sup, fourier_sup, R_1, R_2, phi_freq, radi_freq, Imin_small, 
                         Imax_small, fourier_enl=[(-20, 20), (-0.06, 0.06)])
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back where a gaussian is subtracted from the center
        f_ap_sup, _, _ = aperture_flux_warped(warped_sup, warped_shape, R_1, 
                                              aspect_rad, model_planet)
        print("The aperture flux of the model planet without (only at radial freq=0) spyders is: ", f_ap_sub)
        print("This corresponds to ", round(100/f_ap_f*f_ap_sup, 3), " %")
        
        # The function does everything done above -> summarize the essential 
        # things which work (so only central frequencies)
        #_, _, _, _ = suppress_subtraction(int1, R_1, R_2, plot=True)
        
     
    
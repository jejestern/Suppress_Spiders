#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-03-03
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
from filter_functions import e_func, Gaussian1D
import aotools



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

# Create a list in which we can save the information of the fft of the warped image
aper_origin = []
aper_warped = []
aper_flat = []
aper_cfreq = []
aper_wspyd = []

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
    
        
        # Define the corresponding polar coordinates to the x-y coordinates
        r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))
        mask_r = radius_mask(r_array, (R_1, R_2))
        mask_phi = angle_mask(phi_array, (0, 2*np.pi))
        mask = mask_r & mask_phi
        
        # PSF
        zeros = np.zeros((x_len, y_len))
        mask_psf = aotools.circle(64,128)-aotools.circle(16, 128)
        zeros[:128,:128] = mask_psf 
        psf = aotools.ft2(zeros, delta=1./128.,)
        psf = abs(psf)
        
        psf_pos = (300, 800)
        psf[700:900, 200:400] = psf[512-100:512+100, 512-100:512+100]
        psf[512-100:512+100, 512-100:512+100] = psf[0:200, 0:200]
        
        #int1 = int1 + psf
        
        ## Computation of the aperture flux of the ghost
        model_planet = gh_pos[1] #psf_pos  
        f_ap_im, ap_im, annu_im = aperture_flux_image(int1, model_planet)
        print("The aperture flux of the model planet in the original image is: ", f_ap_im)
        aper_origin.append(f_ap_im)
        
        plt.imshow(int1*mask, origin='lower', cmap='gray', vmin=0, vmax=Imax)
        ap_im.plot(color ='r', lw=1.0)
        annu_im.plot(color ='#0547f9', lw=1.0)
        plt.colorbar()
        plt.tight_layout()
        #plt.savefig("interpolation/HDimg_R150_R300.pdf")
        plt.show()
        

        # Warp the image to the r-phi Plane    
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_shape = warped.shape
        warped = warped.T
        
        # Define the radial and angular coordinates
        radi = np.arange(warped_shape[1])
        phis = np.arange(warped_shape[0])/warped_shape[0]*2*np.pi
        middle = int(R_1 + (R_2 - R_1)/2)
            
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
        aper_warped.append(f_ap_w)
        

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
        aper_flat.append(f_ap_f)

        ## Plot the output and its fft
        fourier = np.fft.fftshift(np.fft.fft2(warped))
        fourier_real = fourier.real
      
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped, origin='lower', aspect=aspect_rad, vmin=Imin_small, 
                   vmax= Imax_small, extent=[0, 2*np.pi, R_1, R_2])
        ap_f_draw.plot(color ='r', lw=1.0)
        annu_f_draw.plot(color ='#0547f9', lw=1.0)
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()

        plt.subplot(212)
        plt.imshow(abs(fourier), origin='lower', cmap='gray', norm=LogNorm(vmin=1),
                   aspect=aspect_freq, 
                   extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
 
        plt.tight_layout()
        plt.savefig("suppression/HDflatten_R254_R454_-1to1.pdf")
        plt.show()
        
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
        plt.savefig("suppression/rad0.pdf")
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
        plt.savefig("suppression/ang0.pdf")
        plt.show()
        """
        ##Subtract the background from everything
        b = int((R_2-R_1)/2)
        while b > 0:
            background = np.mean(abs(fourier[b, :int(len(phis)/8)]))
            print(background)
            fourier[b, :] = fourier[b, :] / background
            if b != int((R_2-R_1)/2):
                fourier[-b, :] = fourier[-b, :] / background
            b -= 1
        """
        
        ## Set everything outside a specific area to zero
        neg_r = 40
        neg_a = 300
        """
        q = 1 #10**(-3)
        fourier[:middle-R_1-neg_r+1, :] = fourier[:middle-R_1-neg_r+1, :]*q
        fourier[middle-R_1+neg_r:, :] = fourier[middle-R_1+neg_r:, :]*q
        fourier[:, :int(len(phis)/2)-neg_a+1] = fourier[:,  :int(len(phis)/2)-neg_a+1]*q
        fourier[:, int(len(phis)/2)+neg_a:] = fourier[:, int(len(phis)/2)+neg_a:]*q
        """
        # Inverse FFT after setting the edges to a smaaaall value
        warped_back = np.fft.ifft2(np.fft.ifftshift(fourier)).real

        ## Computation of the aperture flux of the model planet after background=0
        f_ap_b, _, _ = aperture_flux_warped(warped_back, warped_shape, 
                                                              R_1, aspect_rad, 
                                                              model_planet)
        print("The aperture flux of the model planet concidering only central frequencies: ", f_ap_b)
        print("This corresponds to ", round(100/f_ap_f*f_ap_b, 3), " %")
        aper_flat.append(f_ap_b)
        
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped_back, origin='lower', aspect=aspect_rad, vmin=Imin_small, 
                   vmax= Imax_small, extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier), origin='lower', cmap='gray', 
                   norm=LogNorm(vmin=1), aspect=aspect_freq, 
                   extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        #plt.imshow(fourier.imag, origin='lower', cmap='gray', aspect=aspect_freq, vmin=-100, 
        #           vmax= 100, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
        
        plt.tight_layout()
        #plt.savefig("interpolation/HDflatten_R150_R300_4.pdf")
        plt.show()
      
        
        # Subtract Gaussian 
        ### Take out some structure via fft: SUBTRACTION of gaussian of the 
        ### center frequencyies (radial)
        W = np.arange(40, 80)
        
        apertures_cen = []
        apertures_W = []
        apertures_I = []
        
        # Create the Gaussian profile
        w_g = 55 #68 # Width of Gaussian profile
        W_G = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        
        for w in W:
            gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, 10**4)
        
            #Subtract it
            fft_w = fourier.copy()
            spid_center = fft_w[middle-R_1, :].copy()
            spid_center[int(len(phis)/2)-w:int(len(phis)/2)+w] = spid_center[int(len(phis)/2)-
                                                                             w:int(len(phis)/2)
                                                                             +w]/gauss[int(len(phis)/2)-w:int(len(phis)/2)+w]
            fft_w[middle-R_1, :] = spid_center
            warped_back = abs(np.fft.ifft2(fft_w))
            f_w, _, _ = aperture_flux_warped(warped_back,warped_shape, R_1,
                                                  aspect_rad, model_planet)
            apertures_W.append(f_w)
        
        aperture_ini = lambda x: f_ap_f + x*0
        plt.figure()
        plt.plot(W/warped_shape[0]*phi_freq[-1]*2, apertures_W, 'o', label='Aperture flux of ghost 2')
        plt.plot(W/warped_shape[0]*phi_freq[-1]*2, aperture_ini(W), 'orange', label='Initial aperture flux of ghost 2')
        plt.xlabel(r"Subtraction width [$\frac{1}{\mathrm{rad}}$]")
        plt.legend()
        plt.savefig("suppression/rad0_diffsubwidths.pdf")
        plt.show()
        
        print(W[np.where(apertures_W == max(apertures_W))[0]])
        print(W[np.where(apertures_W == max(apertures_W))[0]]/warped_shape[0]*phi_freq[-1]*2)
        
        w = 61 # With this value we have the smallest aperture flux loss (ghost)
        """
        for i in W_G:
            gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), i, 10**4)
        
            #Subtract it
            fft_wg = fourier.copy()
            spid_center = fft_wg[middle-R_1, :].copy()
            spid_center[int(len(phis)/2)-w:int(len(phis)/2)+w] = spid_center[int(len(phis)/2)-
                                                                             w:int(len(phis)/2)
                                                                             +w]/gauss[int(len(phis)/2)-w:int(len(phis)/2)+w]
            fft_wg[middle-R_1, :] = spid_center
            warped_back = abs(np.fft.ifft2(fft_wg))
            f_gauss, _, _ = aperture_flux_warped(warped_back,warped_shape, R_1,
                                                  aspect_rad, model_planet)
            apertures_cen.append(f_gauss)
            
        
        plt.figure()
        plt.plot(W_G, apertures_cen, 'o')
        #plt.ylim((142, 146))
        plt.show()
        
        Intens = np.linspace(10**3, 10**5, 15)
        for i in Intens:
            gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, i)
        
            #Subtract it
            fft_i = fourier.copy()
            spid_center = fft_i[middle-R_1, :].copy()
            spid_center[int(len(phis)/2)-w:int(len(phis)/2)+w] = spid_center[int(len(phis)/2)-
                                                                             w:int(len(phis)/2)
                                                                             +w]/gauss[int(len(phis)/2)-w:int(len(phis)/2)+w]
            fft_i[middle-R_1, :] = spid_center
            warped_back = abs(np.fft.ifft2(fft_i))
            f_I, _, _ = aperture_flux_warped(warped_back,warped_shape, R_1,
                                                  aspect_rad, model_planet)
            apertures_I.append(f_I)
            
        
        plt.figure()
        plt.plot(Intens, apertures_I, 'o')
        #plt.ylim((142, 146))
        plt.show()
        """
        
        # Now the final subtraction = division by the gaussian
        I_g = 9.918*10**3
        gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, I_g)
        #gauss1 = Gaussian1D(phi_freq.copy(), int(len(phis)/2), 65, 2500) 
        #gauss2 = Gaussian1D(phi_freq.copy(), int(len(phis)/2), 15, 10000)
        #gauss = gauss1+gauss2
        
        #Subtract it
        spid_center = fourier[middle-R_1, :].copy()
        fourier[middle-R_1, int(len(phis)/2)-w:int(len(phis)/2)+w] = spid_center[int(len(phis)/2)-
                                                                         w:int(len(phis)/2)
                                                                         +w]/gauss[int(len(phis)/2)-w:int(len(phis)/2)+w]
     
        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spid_center), label="spid")
        plt.semilogy(phi_freq, abs(gauss)+0.001, label="gauss $\sigma$ = %.i" %(w_g))
        plt.xlim((-50, 50))
        plt.ylim((10**(-1), 10**(5)))
        plt.title("FFT ratio of beam images")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend()
        plt.show()
    
        fourier_real = fourier.real
        warped_back = np.fft.ifft2(np.fft.ifftshift(fourier)).real
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back where a gaussian is subtracted from the center
        f_ap_fft, ap_fft_draw, annu_fft_draw = aperture_flux_warped(warped_back, 
                                                                    warped_shape, R_1, 
                                                                    aspect_rad, 
                                                                    model_planet)
        print("The aperture flux of the model planet without (only at radial freq=0) spyders is: ", f_ap_fft)
        print("This corresponds to ", round(100/f_ap_f*f_ap_fft, 3), " %")
        aper_cfreq.append(f_ap_fft)
        
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped_back, origin='lower', aspect=aspect_rad, vmin=Imin_small, 
                   vmax= Imax_small, extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier), origin='lower', cmap='gray', 
                   norm=LogNorm(vmin=1), aspect=aspect_freq, 
                   extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("suppression/HDcentralfreq_R254_R454_-1to1.pdf")
        plt.show()
         
        ######################################################################
        ######### Frequency suppression also in radial direction #############
        ######################################################################
        
        # Ratio in radial direction in order to make a Gaussian subtraction of all 
        # frequencies in radial direction (also the larger ones)
        cen_r = int((R_2-R_1)/2)
        ratio_gauss = np.sum(abs(fourier[cen_r+1:cen_r+neg_r, int(len(phis)/2)-w:
                                         int(len(phis)/2)-1]), axis=1)/np.sum(abs(spid_center[
                                             int(len(phis)/2)-w:int(len(phis)/2)-1]))        
        #print("Ratio for the small gaussian: ", ratio_gauss)
        
        # We define the Gauss for the non-zero radial frequency, which has a 
        # smaller width 
        w_s = 40
        gauss_s = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_s, 0.84*I_g)
       
        y = int((R_2-R_1)/2) + 1
        y_g = 0
        plt.figure(figsize=(8, 16*aspect_value))
        while y > cen_r - neg_r:
            plt.semilogy(phi_freq, abs(fourier[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
            plt.semilogy(phi_freq, gauss_s*ratio_gauss[y_g], label="Gaussian $\sigma$ = %.i" %(w_g))
            if y == int((R_2-R_1)/2):
                y -= 1
                y_g += 1
            elif abs(radi_freq[y]) < 0.02:
                y -= 5
                y_g += 5
            else:
                y -= 20
                y_g += 20
        #plt.semilogy(phi_freq, abs(spid_center), label="spid")
        #plt.semilogy(phi_freq, abs(gauss), label="gauss")
        plt.xlim((-30, 30))
        plt.ylim((10**(1), 3.5*10**(4)))
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
        
        r_n = cen_r - 1
        r_p = cen_r + 1
        ratio_i = 0
        while r_n > cen_r - 3:
            w_s = int(0.84*w * ratio_gauss[ratio_i])
            fourier[r_n, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s] = fourier[
                r_n, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s]/(
                    gauss_s[int(len(phis)/2)-w_s:int(len(phis)/2)+w_s]*ratio_gauss[ratio_i])
            fourier[r_p, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s] = fourier[
                r_p, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s]/(
                    gauss[int(len(phis)/2)-w_s:int(len(phis)/2)+w_s]*ratio_gauss[ratio_i])
            r_n -= 1
            r_p += 1
            ratio_i += 1
            
        warped_back = np.fft.ifft2(np.fft.ifftshift(fourier)).real 
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back image where the spyders are taken away
        f_ap_fft, ap_fft_draw, annu_fft_draw = aperture_flux_warped(warped_back, 
                                                                    warped_shape, R_1, 
                                                                    aspect_value, 
                                                                    model_planet)
        print("The aperture flux of the model planet without spyders is: ", f_ap_fft)
        print("This corresponds to ", round(100/f_ap_f*f_ap_fft, 3), " %")
        aper_wspyd.append(f_ap_fft)
        
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped_back, origin='lower', aspect=aspect_rad, vmin=Imin_small, 
                   vmax= Imax_small, extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.subplot(212)
        plt.imshow(abs(fourier), origin='lower', cmap='gray', 
                   norm=LogNorm(vmin=1), aspect=aspect_freq, 
                   extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
        plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
        plt.ylim((-0.5, 0.5))
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("suppression/HDcentralfreq_R254_R454_-1to1.pdf")
        plt.show()

        
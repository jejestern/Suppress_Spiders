#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Includes functions used for the suppression. As a function which plots the image
and the fourier transform in one plot. And a function which divides the FFT by
a Gaussian in order to suppress the spiders.

Created on 2022-03-22
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformations_functions import to_rphi_plane, flatten_img
from filter_functions import Gaussian1D
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def fourier_plotting(img, fourier, R_1, R_2, phi_freq, radi_freq, Imin, Imax, 
                     fourier_enl=None, savefig=None):
    """

    Parameters
    ----------
    img : 2D array
        Image which we want to plot
    fourier : 2D array
        FFT of the image
    R_1 : int
        Minimal radius considered in the radius range
    R_2 : int
        Maximal radius considered in the radius range
    phi_freq : 1D array
        The frequency range in angular direction
    radi_freq : 1D array
        The frequency range in radial direction
    Imin : float
        Minimal intensity plotted of the image
    Imax : float
        Maximal intensity plotted of the image
    fourier_enl : [x_lim, y_lim], optional
        If we only want to look at a specific region of the FFT. The default is None.
    savefig : str, optional
        The title of the pdf, if we want to save the image. The default is None.

    Returns
    -------
    int
        No return, just plots the image.

    """
    
    shape = img.shape
    
    # The different aspect ratios used for plotting
    aspect_value = (360/shape[1])/((R_2-R_1)/shape[0])
    aspect_rad = (2*np.pi/shape[1])/((R_2-R_1)/shape[0])
    aspect_freq = ((phi_freq[-1]-phi_freq[0])/shape[1])/((radi_freq[-1]-radi_freq[0])/shape[0])
    
    # Plot the image and its fft
    plt.figure(figsize=(8, 16*aspect_value))

    plt.subplot(211)
    plt.imshow(img, origin='lower', aspect=aspect_rad, vmin=Imin, 
               vmax= Imax, extent=[0, 2*np.pi, R_1, R_2])
    plt.xlabel(r'$\varphi$ [rad]')
    plt.ylabel('Radius')
    plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
               [r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    #plt.ylim((300, 400))
    plt.colorbar()
    
    plt.subplot(212)
    plt.imshow(abs(fourier), origin='lower', cmap='gray', norm=LogNorm(vmin=1),
               aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], 
                                           radi_freq[-1]])
    plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
    plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
    plt.ylim((-0.5, 0.5))
    if fourier_enl != None:
        plt.xlim(fourier_enl[0])
        plt.ylim(fourier_enl[1])
    plt.colorbar()
 
    if savefig != None:
        plt.savefig(savefig)
    plt.tight_layout()
    plt.show()
    
    return 0

    
def suppress_division(img, R_1, R_2, plot=False):
    
    # Calculation of the image shapes
    x_len, y_len = img.shape
    
    # Warp the image to the r-phi Plane    
    warped = to_rphi_plane(img, (x_len, y_len), R_1, R_2)
    warped_shape = warped.shape
    warped = warped.T
    
    # Define the radial and angular coordinates
    radi = np.arange(warped_shape[1])
    phis = np.arange(warped_shape[0])/warped_shape[0]*2*np.pi
    cen_r = int((R_2-R_1)/2)
    
    # The physical frequencies
    phi_freq = np.fft.fftfreq(warped_shape[0], d=2*np.pi/warped_shape[0])
    phi_freq = np.fft.fftshift(phi_freq)
    radi_freq = np.fft.fftfreq(warped_shape[1])
    radi_freq = np.fft.fftshift(radi_freq)
    
    # First we take out the intensity change in radial direction due to the
    # star in the center, by using an exponential fit function.
    flatten = flatten_img(warped, warped_shape, radi)
 
    # Fourier transform the flatten image
    fourier_f = np.fft.fftshift(np.fft.fft2(flatten))
         
    ################ Frequency suppression: central freq #####################
    # Central frequencies (radial) are divided by a Gaussian profile
    # Create the Gaussian profile
    sigma_g = 55 # Width of Gaussian
    I_g = 9.918*10**3  # Intensity of the Gaussian
        
    gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), sigma_g, I_g)
    
    # Division
    w = 57  # angular division width
    fourier_c = fourier_f.copy()
    fourier_c[cen_r, int(len(phis)/2)-w:
              int(len(phis)/2)+w] = fourier_f[cen_r, int(len(phis)/2)-w:
                                              int(len(phis)/2)+w]/gauss[int(len(phis)/2)-w:
                                                                        int(len(phis)/2)+w]
    
                                                                        # Inverse Fourier transform back                                                                  
    flatten_c = np.fft.ifft2(np.fft.ifftshift(fourier_c)).real
     
    ########### Frequency suppression also in radial direction ###############
    # Low frequencies are divided by a Gaussian
    # Create the Gaussian profile for this division
    sigma_s = 40 # Width of this Gaussian
    I_s = 0.84*I_g
    
    gauss_s = Gaussian1D(phi_freq.copy(), int(len(phis)/2), sigma_s, I_s)

    # Ratio in radial direction of the low frequencies, in order to adapt the 
    # Gaussian profile and the angular division width to the radial frequency,
    # mainly its intensity
    neg_r = 40 # angular with considered
    ratio_gauss = np.sum(abs(fourier_f[cen_r+1:cen_r+neg_r, int(len(phis)/2)-w:
                                       int(len(phis)/2)-1]), axis=1)/np.sum(abs(fourier_f[
                                           cen_r,int(len(phis)/2)-w:int(len(phis)/2)-1])) 
    # Division           
    r_n = cen_r - 1
    r_p = cen_r + 1
    ratio_i = 0
    h = 3
    fourier_l = fourier_c.copy()
    while r_n > cen_r - h:
        w = int(45* ratio_gauss[ratio_i])
        fourier_l[r_n, int(len(phis)/2)-w:int(len(phis)/2)+w] = fourier_l[
            r_n, int(len(phis)/2)-w:int(len(phis)/2)+w]/(
                gauss_s[int(len(phis)/2)-w:int(len(phis)/2)+w]*ratio_gauss[ratio_i])
        fourier_l[r_p, int(len(phis)/2)-w:int(len(phis)/2)+w] = fourier_l[
            r_p, int(len(phis)/2)-w:int(len(phis)/2)+w]/(
                gauss[int(len(phis)/2)-w:int(len(phis)/2)+w]*ratio_gauss[ratio_i])
        r_n -= 1
        r_p += 1
        ratio_i += 1
    
    # Inverse Fourier transform back
    flatten_l = np.fft.ifft2(np.fft.ifftshift(fourier_l)).real 
    

    if plot == True:
        # Choose the intensity
        Imax = 0.5
        Imin = -0.5
            
        fourier_plotting(flatten, fourier_f, R_1, R_2, phi_freq, radi_freq, 
                         Imin, Imax)
        
        fourier_plotting(flatten_c, fourier_c, R_1, R_2, phi_freq, radi_freq, 
                         Imin, Imax, fourier_enl=[(-20, 20), (-0.06, 0.06)])
        
        fourier_plotting(flatten_l, fourier_l, R_1, R_2, phi_freq, radi_freq, 
                         Imin, Imax, fourier_enl=[(-20, 20), (-0.06, 0.06)])
        
        
    return warped_shape, warped, flatten, flatten_c, flatten_l

def suppress_subtraction(img, R_1, R_2, plot=False):
    
    # Calculation of the image shapes
    x_len, y_len = img.shape
    
    # Warp the image to the r-phi Plane    
    warped = to_rphi_plane(img, (x_len, y_len), R_1, R_2)
    warped_shape = warped.shape
    warped = warped.T
    
    # Define the radial and angular coordinates
    radi = np.arange(warped_shape[1])
    phis = np.arange(warped_shape[0])/warped_shape[0]*2*np.pi
    cen_r = int((R_2-R_1)/2)  # Central radius position
    cen_phi = int(warped_shape[0]/2) # central angle position
    
    # The physical frequencies
    phi_freq = np.fft.fftfreq(warped_shape[0], d=2*np.pi/warped_shape[0])
    phi_freq = np.fft.fftshift(phi_freq)
    radi_freq = np.fft.fftfreq(warped_shape[1])
    radi_freq = np.fft.fftshift(radi_freq)
    
    # First we take out the intensity change in radial direction due to the
    # star in the center, by using an exponential fit function.
    flatten = flatten_img(warped, warped_shape, radi)
 
    # Fourier transform the flatten image
    fourier_f = np.fft.fftshift(np.fft.fft2(flatten))
    
    ################ Frequency subtraction: central freq #####################
    
    # We search for the positions of the spiders
    ## Plot the total intensity of the image along the angular axis
    phi_trend = np.sum(flatten, axis = 0)/warped_shape[0]
        
    ## The position of the spider, we know the minimal distance between the 
    ## spiders from observations
    min_dist = 1.36 # In radians
    spos = find_peaks(phi_trend, distance=min_dist/(2*np.pi)*warped_shape[0])[0]
    
    # We search for the width and the maximal intensity of the spiders around 
    # the central radius to simulate the spiders along this axis
    thick_cen = 10  # The radial range around the cen_r considered
    flatten_center = flatten[cen_r -thick_cen:cen_r+thick_cen, :]/(2*thick_cen)
    phi_center = np.sum(flatten_center, axis = 0)
        
    ## We fit a smoothed gaussian (1D) to the spiders
    width = []  # Resulting widths
    g = 0  # This will be the simulated spiders along cen_r
    for i in range(4):
        n = 15  # The range around the spider in angular direction considered for fitting
        phis_n = phis[spos[i]-n:spos[i]+n].copy()
        phi_center_n = phi_center[spos[i]-n:spos[i]+n].copy()
        
        # Fitting
        parameters, covariance = curve_fit(Gaussian1D, phis_n.copy(), phi_center_n.copy())
        width.append(parameters[1])
        fit_y = Gaussian1D(phis.copy(), parameters[0]+spos[i]-n, parameters[1], parameters[2])
        
        # Add the Gaussian fit of this spider to the others
        g += fit_y
    
    # Fourier transform of the simulated 1D spiders
    fft_g = np.fft.fftshift(np.fft.fft(g))
    
    # We want to know approx how much of the total intensity (=central frequency) 
    # is caused by the spiders
    I_tot = 0
    for i in range(len(width)):
        I_tot += np.sum(flatten[:, spos[i]-int(width[i]):spos[i]+int(width[i])])
    
    # Factor which describes the intensity difference between 1D and reality 
    # caused by the spiders
    fac = I_tot/fft_g[cen_phi]

    # Subtraction along central radial frequencies
    fourier_sub = fourier_f.copy()
    spid_center = fourier_f[cen_r, :].copy()
    spid_center = spid_center - fft_g*fac
    fourier_sub[cen_r, :] = spid_center
    
    # We transform the fft back to the r-phi plane -> image where the spiders
    # are subtracted (at least in the middle)
    flatten_sub = np.fft.ifft2(np.fft.ifftshift(fourier_sub)).real
    
    if plot == True:
        # Choose the intensity
        Imax = 0.5
        Imin = -0.5
        
        # We want to plot the real data and the simulation of the spiders and its fft
        fig1, ax1 = plt.subplots(2, 1, figsize=(8, 8))  
        ax1[0].plot(phis, phi_center, label="Mean intensity along central radii")
        ax1[0].plot(phis, g, 'orange',
                    label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
                    %(width[0]/warped_shape[0]*2*np.pi, width[1]/warped_shape[0]*2*np.pi, 
                      width[2]/warped_shape[0]*2*np.pi, width[3]/warped_shape[0]*2*np.pi))
        ax1[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                                 r'$3\pi/2$', r'$2\pi$'])
        ax1[0].set_xlabel(r'$\varphi$ [rad]')
        ax1[0].legend(loc='lower right')
        
        ax1[1].plot(phi_freq, fft_g.real, 'orange', label="FFT of 1D spider simulation")
        ax1[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        ax1[1].set_xlim((-20, 20))
        ax1[1].legend()
        #plt.savefig("fourier/Gaussian_fourdiffspyders.pdf")
        plt.show()
        
        ## Plot along the central radial frequencies: before and after subtraction
        plt.figure(figsize=(8, 4))
        plt.plot(phi_freq, fourier_f[cen_r, :].real, label ="radial freq. = %.2f" %(radi_freq[cen_r]))
        plt.plot(phi_freq, fourier_sub[cen_r, :].real, label ="Spider subtracted" %(radi_freq[cen_r]))
        plt.xlim((-20, 20))
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        #plt.savefig("suppression/rad0.pdf")
        plt.show()
        
        # Plot the flattened image and its fft
        fourier_plotting(flatten, fourier_f, R_1, R_2, phi_freq, radi_freq, 
                         Imin, Imax)  
        
        ## Plot the subtracted image and its fft
        fourier_plotting(flatten_sub, fourier_sub, R_1, R_2, phi_freq, radi_freq, 
                         Imin, Imax, fourier_enl=[(-20, 20), (-0.06, 0.06)])
   
    return warped_shape, warped, flatten, flatten_sub

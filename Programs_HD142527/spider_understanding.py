#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-02-23
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from matplotlib.colors import LogNorm
from transformations_functions import to_rphi_plane, xy_to_rphi
from scipy.optimize import curve_fit
from filter_functions import gaussianBeam, gaussianSpyder, Gaussian1D, e_func
import aotools
from aperture_fluxes import aperture_flux_image, aperture_flux_warped



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

aspect_value = (360/warp_shape[0])/((R_2-R_1)/warp_shape[1])
aspect_rad = (2*np.pi/warp_shape[0])/((R_2-R_1)/warp_shape[1])


radi = np.arange(warp_shape[1])
phis = np.arange(warp_shape[0])/warp_shape[0]*2*np.pi
cen_r = int((R_2-R_1)/2)
cen_phi = int(warp_shape[0]/2)

# Create the axis labels for the fft image
phi_freq = np.fft.fftfreq(warp_shape[0], d=2*np.pi/warp_shape[0])
phi_freq = np.fft.fftshift(phi_freq)
radi_freq = np.fft.fftfreq(warp_shape[1])
radi_freq = np.fft.fftshift(radi_freq)

aspect_freq = ((phi_freq[-1]-phi_freq[0])/warp_shape[0])/(
    (radi_freq[-1]-radi_freq[0])/warp_shape[1])
    
shift = 50 # We use a small shift, so that the position of the first spyder is 
           # not crossing the image borders
degn = 90/360*warp_shape[0]
spos = [42+shift,  670+shift]   #670+shift
degsym = 180/360*warp_shape[0]


# We create a 1D image which is only non-zero at the positions of the spiders
# We want to investigate the fft of this
sep = np.zeros_like(phi_freq)
sep[spos[0]] = 1
sep[int(spos[0]+degsym)] = 1
sep[spos[1]] = 1
sep[int(spos[1]+degsym)] = 1

fft_sep = np.fft.fftshift(np.fft.fft(sep))

# We want to plot  it
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 50*aspect_value))  
ax1[0].plot(phis, sep, 
            label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
            %(5/warp_shape[0]*2*np.pi, 10/warp_shape[0]*2*np.pi, 8/warp_shape[0]*2*np.pi, 
              6/warp_shape[0]*2*np.pi))
ax1[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax1[0].set_xlabel(r'$\varphi$ [rad]')
ax1[0].legend(loc='upper right')

ax1[1].plot(phi_freq, abs(fft_sep), label="FFT")
ax1[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
#ax1[1].set_ylim((10**(-3), 4*10**(4)))
ax1[1].set_xlim((-20, 20))
ax1[1].legend()
#plt.savefig("fourier/Gaussian_fourdiffspyders.pdf")
plt.show()


# We insert smoothed gaussian (1D) at the positions of the spiders with 
# the same width and intensity
g1 = Gaussian1D(phi_freq.copy(), spos[0], 5, 1.7)
g2 = Gaussian1D(phi_freq.copy(), spos[1], 10, 2.5)
g3 = Gaussian1D(phi_freq.copy(), spos[0]+degsym, 8, 2.1)
g4 = Gaussian1D(phi_freq.copy(), spos[1]+degsym, 6, 0.8)

g = g1 + g2 + g3 + g4
fft_g = np.fft.fftshift(np.fft.fft(g))

# We want to plot  it
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 50*aspect_value))  
ax1[0].plot(phis, g, 
            label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
            %(5/warp_shape[0]*2*np.pi, 10/warp_shape[0]*2*np.pi, 8/warp_shape[0]*2*np.pi, 
              6/warp_shape[0]*2*np.pi))
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

# We insert smoothed (gaussian) beams at the positions of the spiders with 
# the same width and intensity
beamG1 = gaussianBeam(warp_or.copy(), spos[0], 5, 1.7)
beamG2 = gaussianBeam(warp_or.copy(), spos[1], 10, 2.5)
beamG3 = gaussianBeam(warp_or.copy(), spos[0]+degsym, 8, 2.1)
beamG4 = gaussianBeam(warp_or.copy(), spos[1]+degsym, 6, 0.8)

beamG = beamG1 + beamG2 + beamG3 + beamG4
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))

# Factor which describes the intensity difference between 1D and 2D
fac = fft_beamG[cen_r, cen_phi]/fft_g[cen_phi]
fac_sep = fft_beamG[cen_r, cen_phi]/fft_sep[cen_phi]

# We want to plot  the fft
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 50*aspect_value))  
ax1[0].plot(phis, beamG[cen_r, :], 
            label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
            %(5/warp_shape[0]*2*np.pi, 10/warp_shape[0]*2*np.pi, 8/warp_shape[0]*2*np.pi, 
              6/warp_shape[0]*2*np.pi))
ax1[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax1[0].set_xlabel(r'$\varphi$ [rad]')
ax1[0].legend(loc='upper right')

ax1[1].plot(phi_freq, fft_beamG[cen_r, :].real, label="FFT")
ax1[1].plot(phi_freq, fft_g*fac.real, label="FFT of previous normalised")
ax1[1].plot(phi_freq, fft_sep.real*fac_sep, label="seperation")
ax1[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
#ax1[1].set_ylim((10**(-3), 4*10**(4)))
ax1[1].set_xlim((-20, 20))
ax1[1].legend()
#plt.savefig("fourier/Gaussian_fourdiffspyders.pdf")
plt.show()



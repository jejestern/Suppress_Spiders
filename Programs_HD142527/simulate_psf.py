#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:48:19 2022

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformations_functions import to_rphi_plane, xy_to_rphi
import aotools

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianBeam(base, y_position, x_position, D0):
    rows, cols = base.shape
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x), (y_position, x_position))**2)/(2*(D0**2))))
    return base


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


# Create the axis labels for the fft image
phi_freq = np.fft.fftfreq(warp_shape[0], d=2*np.pi/warp_shape[0])
phi_freq = np.fft.fftshift(phi_freq)
radi_freq = np.fft.fftfreq(warp_shape[1])
radi_freq = np.fft.fftshift(radi_freq)

aspect_freq = ((phi_freq[-1]-phi_freq[0])/warp_shape[0])/(
    (radi_freq[-1]-radi_freq[0])/warp_shape[1])


# We insert smoothed (gaussian) beams at the positions of the spyders
position_x = 400
position_y = 90
gauss = gaussianBeam(warp_or.copy(), position_y, position_x, 10)

  
# Fourier transform the warped image with beams
fft_gauss = np.fft.fftshift(np.fft.fft2(gauss))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(gauss, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_gauss + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()

# PSF
mask = aotools.circle(64,128)-aotools.circle(16, 128)
zeros[:128,:128] = mask
psf = aotools.ft2(zeros, delta=1./128.,)
psf = abs(psf)

psf[700:900, 200:400] = psf[512-100:512+100, 512-100:512+100]
psf[512-100:512+100, 512-100:512+100] = psf[0:200, 0:200]

r_pos, phi_pos = xy_to_rphi(300-512, 800-512)
r_pos = round(r_pos)
phi_pos = phi_pos*180/np.pi

# Warp the image into the r-phi plane
warp_psf = to_rphi_plane(psf, (x_len, y_len), R_1, R_2)
warp_psf_or = warp_psf.T

# Fourier transform the warped image with beams
fft_psf = np.fft.fftshift(np.fft.fft2(warp_psf_or))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(warp_psf_or, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
      
plt.subplot(212)
plt.imshow(abs(fft_psf + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
plt.savefig("fourier/PSF_fourier.pdf")
plt.show()



plt.figure(figsize=(8, 25*aspect_value))
plt.plot(phis[int(len(phis)/16):int(len(phis)/4)], gauss[position_y, int(len(phis)/16):int(len(phis)/4)], label="Gaussian")
plt.plot(phis[int(len(phis)/16):int(len(phis)/4)], warp_psf_or[r_pos-R_1, int(len(phis)/16):int(len(phis)/4)], label="PSF")
#plt.title("Horizontal cut")
plt.xticks([np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2], [r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
plt.xlabel(r'$\varphi$ [rad]')
plt.legend()
plt.savefig("fourier/PSF_cut_image.pdf")
plt.show()

plt.figure(figsize=(8, 25*aspect_value))
plt.semilogy(phi_freq, abs(fft_gauss[middle-R_1, :] + 0.0001), label="Gaussian")
plt.semilogy(phi_freq, abs(fft_psf[middle-R_1, :] + 0.0001), label="PSF")
#plt.ylim((10**(-1), 10**(5)))
#plt.title("FFT")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend()
plt.savefig("fourier/PSF_cut_fourier.pdf")
plt.show()

plt.figure(figsize=(8, 16*aspect_value))
plt.plot(radi+R_1, gauss[:, position_x], label="Gaussian")
plt.plot(radi+R_1, warp_psf_or[:, round(phi_pos/360*warp_shape[0])], label="PSF")
plt.title("Vertical cut through")
plt.xlabel('Radius')
plt.legend()
plt.show()

plt.figure(figsize=(8, 16*aspect_value))
plt.semilogy(radi_freq, abs(fft_gauss[:, int(len(phis)/2)] + 0.0001), label="Gaussian")
plt.semilogy(radi_freq, abs(fft_psf[:, int(len(phis)/2)] + 0.0001), label="PSF")
#plt.ylim((10**(-1), 10**(5)))
#plt.title("FFT")
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend()
plt.show()
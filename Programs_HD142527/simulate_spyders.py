#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:21:00 2022

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformations_functions import to_rphi_plane

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianBeam(base, x_position, D0):
    rows, cols = base.shape
    for x in range(cols):
        base[:,x] = np.exp(((-distance((0,x), (0, x_position))**2)/(2*(D0**2))))
    return base

def gaussianSpyder(base, x_position, D0):
    rows, cols = base.shape
    y = 0
    while y < rows: 
        if D0 > 1:
            D0 -= 0.08
            
        for x in range(cols):
            base[y,x] = np.exp(((-distance((y,x), (y, x_position))**2)/(2*(D0**2))))
            
        y += 1
            
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

# We insert beams at the positions of the spyders
beams = warp_or.copy()
beams[:, 31:53] = 1
beams[:, 654:686] = 1
beams[:, 1143:1167] = 1
beams[:, 1767:1791] = 1
        
# Fourier transform the warped image with beams
fft_beams = np.fft.fftshift(np.fft.fft2(beams))

# Create the axis labels for the fft image
xf = np.fft.fftfreq(360, 360/warp_shape[0])
xf = np.fft.fftshift(xf)
yf = np.fft.fftfreq(R_2-R_1, (R_2-R_1)/warp_shape[1])
yf = np.fft.fftshift(yf)
print(xf)      
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(beams, origin='lower', aspect=aspect_value, vmin=0, vmax=Imax, 
           extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()
        
plt.subplot(212)
#plt.imshow(abs(fft_beams + 0.0001), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
 #          aspect=aspect_value, extent=[0, 360, R_1, R_2])
plt.imshow(abs(fft_beams + 0.0001), origin='lower', cmap='gray', norm=LogNorm(vmin=1), 
           aspect=aspect_value, extent=[xf[0], xf[-1], yf[0], yf[-1]])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
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
plt.imshow(beamG, origin='lower', aspect=aspect_value, vmin=0, vmax=Imax, 
           extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_beamG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_value, extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()


radi = np.arange(warp_shape[1])
phis = np.arange(warp_shape[0])
middle = int(R_1 + (R_2 - R_1)/2)

plt.figure(figsize=(8, 16*aspect_value))
plt.plot(phis, beams[middle-R_1, :], label="beams")
plt.plot(phis, beamG[middle-R_1, :], label="Gaussian beams")
plt.title("Horizontal cut through the beam images")
plt.legend()
plt.show()

plt.figure(figsize=(8, 16*aspect_value))
plt.semilogy(phis, abs(fft_beams[middle-R_1, :] + 0.0001), label="beams")
plt.semilogy(phis, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images")
plt.legend()
plt.show()
"""
# We insert smoothed (gaussian) simulated spyders at the positions of the spyders        
spydG1 = gaussianSpyder(warp_or.copy(), 42, 11)
spydG2 = gaussianSpyder(warp_or.copy(), 670, 16)
spydG3 = gaussianSpyder(warp_or.copy(), 1155, 12)
spydG4 = gaussianSpyder(warp_or.copy(), 1779, 12)

spydG = spydG1 + spydG2 + spydG3 + spydG4
  
# Fourier transform the warped image with beams
fft_spydG = np.fft.fftshift(np.fft.fft2(spydG))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(spydG, origin='lower', aspect=aspect_value, vmin=0, vmax=Imax, 
           extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_spydG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_value, extent=[0, 360, R_1, R_2])
plt.xlabel(r'$\varphi$ [degrees]')
plt.ylabel('Radius')
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()


y = 0            
plt.figure(figsize=(8, 16*aspect_value))
while y < (R_2-R_1): 
    plt.plot(phis, spydG[y, :], label="Simulated spyders")
    y += 15
plt.title("Horizontal cut through the beam images")
plt.legend()
plt.show()

y = 0
plt.figure(figsize=(8, 16*aspect_value))
while y < (R_2-R_1)/2:
    plt.semilogy(phis, abs(fft_spydG[y, :] + 0.0001), label ="radial pos. = %.2f" %(y+R_1))
    y += 15
plt.semilogy(phis, abs(fft_spydG[middle-R_1, :] + 0.0001), label="Centre")
plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images horizontal")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

x = 0
plt.figure()
while x < len(phis)/2:
    plt.semilogy(radi, abs(fft_spydG[:, x] + 0.0001), label ="phi pos. = %.2f" %(x))
    x += 200
plt.semilogy(radi, abs(fft_spydG[:, int(len(phis)/2)] + 0.0001), label="Centre")
plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images horizontal")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
"""
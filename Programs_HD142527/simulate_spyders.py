#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-01-04
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformations_functions import to_rphi_plane
from scipy.optimize import curve_fit
from filter_functions import gaussianBeam, gaussianSpyder


def oneover_x(x, a, b, c):

    return  a * (x-b)**(-1) + c



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
#beamG2 = gaussianBeam(warp_or.copy(), 670, 16)
#beamG3 = gaussianBeam(warp_or.copy(), 1155, 12)
#beamG4 = gaussianBeam(warp_or.copy(), 1779, 12)

beamG = beamG1 #+ beamG2 + beamG3 + beamG4
  
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



plt.figure(figsize=(8, 16*aspect_value))
plt.plot(phis, beams[middle-R_1, :], label="beams")
plt.plot(phis, beamG[middle-R_1, :], label="Gaussian beams")
plt.title("Horizontal cut through the beam images")
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.legend()
plt.show()

plt.figure(figsize=(8, 16*aspect_value))
plt.semilogy(phi_freq, abs(fft_beams[middle-R_1, :] + 0.0001), label="beams")
plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend()
plt.show()

# RATIO of gaussian
w = 60
q = 100
beam_ratio = fft_beams[middle-R_1, :]
beam_ratio[int(len(phis)/2)-w:int(len(phis)/2)+w] = beam_ratio[int(len(phis)/2)
                                                               -w:int(len(phis)/2)
                                                               +w]/fft_beamG[middle-R_1, int(len(phis)/2)-w:int(len(phis)/2)+w]
beam_ratio[:int(len(phis)/2)-w] = beam_ratio[:int(len(phis)/2)-w]/q
beam_ratio[int(len(phis)/2)+w:] = beam_ratio[int(len(phis)/2)+w:]/q

plt.figure(figsize=(8, 16*aspect_value))
plt.semilogy(phi_freq, abs(beam_ratio), label="ratio")
#plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
#plt.xlim((-20, 20))
plt.ylim((10**(-1), 10**(2)))
plt.title("FFT ratio of beam images")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend()
plt.show()

fft_beams[middle-R_1, :] = beam_ratio
fft_back_ratio = abs(np.fft.ifft2(fft_beams))

# Plotting the back transformation
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(abs(fft_beams + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

        
plt.subplot(212)
plt.imshow(fft_back_ratio, origin='lower', aspect=aspect_rad, vmin=0, vmax=0.01, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()

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
plt.imshow(spydG, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_spydG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], 
                                       radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()


y = 0            
plt.figure(figsize=(8, 16*aspect_value))
while y < (R_2-R_1): 
    plt.plot(phis, spydG[y, :], label="at y = %.0f" %(y+R_1))
    y += 50
plt.title("Horizontal cut through the beam images")
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.legend()
plt.show()

y = 0
plt.figure(figsize=(8, 16*aspect_value))
while y < (R_2-R_1)/2:
    plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    y += 20
plt.semilogy(phi_freq, abs(fft_spydG[middle-R_1, :] + 0.0001), label="radial freq. = 0")
plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beam")
plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images horizontal")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


x = 0
plt.figure()
while x < len(phis)/2:
    plt.semilogy(radi_freq, abs(fft_spydG[:, x] + 0.0001), label ="phi pos. = %.1f" %(phi_freq[x]))
    x += 240
plt.semilogy(radi_freq, abs(fft_spydG[:, int(len(phis)/2)] + 0.0001), label="phi pos. = 0")
plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images horizontal")
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

## Fitting an exponential to the horizontal through the center frequency
popt, pcov = curve_fit(oneover_x, radi_freq[middle-R_1+1:] , abs(fft_spydG[middle-R_1+1:, int(len(phis)/2)]))

plt.figure()
plt.semilogy(radi_freq[middle-R_1+1:], abs(fft_spydG[middle-R_1+1:, int(len(phis)/2)]), '.', label="phi pos. = 0")
plt.semilogy(radi_freq[middle-R_1+1:], oneover_x(radi_freq[middle-R_1+1:], *popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images horizontal")
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# RATIO of gaussian
spyd_center = fft_spydG[middle-R_1, :] 
spyd_center[int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_center[int(len(phis)/2)
                                                                 -w:int(len(phis)/2)
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

fft_spydG[middle-R_1, :] = spyd_center
fft_back_spyd_center = abs(np.fft.ifft2(fft_spydG))

# Plotting the back transformation
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(abs(fft_spydG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

        
plt.subplot(212)
plt.imshow(fft_back_spyd_center, origin='lower', aspect=aspect_rad, vmin=0, vmax=0.5, 
           extent=[0, 2*np.pi, R_1, R_2])
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
ratio_gaussian = abs(fft_spydG[middle-R_1, int(len(phis)/2)])/abs(
    fft_spydG[10, int(len(phis)/2)])
small_gaussian = fft_beamG[middle-R_1, :] * ratio_gaussian

y = 0
plt.figure(figsize=(8, 16*aspect_value))
while y < (R_2-R_1)/2:
    plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    y += 20
#plt.semilogy(phi_freq, abs(fft_spydG[middle-R_1, :] + 0.0001), label="radial freq. = 0")
#plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beam")
plt.semilogy(phi_freq, abs(small_gaussian + 0.0001), label="Gaussian beam")
plt.ylim((10**(-1), 10**(3)))
plt.title("FFT of beam images horizontal")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


spyd_low = fft_spydG[:middle-R_1, :] 
spyd_low[:, int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_low[:, int(len(phis)/2)
                                                              -w:int(len(phis)/2)
                                                              +w]/small_gaussian[int(len(phis)/2)-w:int(len(phis)/2)+w]
spyd_high = fft_spydG[middle-R_1:, :] 
spyd_high[:, int(len(phis)/2)-w:int(len(phis)/2)+w] = spyd_high[:, int(len(phis)/2)
                                                                -w:int(len(phis)/2)
                                                                +w]/small_gaussian[int(len(phis)/2)-w:int(len(phis)/2)+w]

#spyd_center[:int(len(phis)/2)-w] = spyd_center[:int(len(phis)/2)-w]/q
#spyd_center[int(len(phis)/2)+w:] = spyd_center[int(len(phis)/2)+w:]/q

plt.figure(figsize=(8, 16*aspect_value))
plt.semilogy(phi_freq, abs(spyd_low[5]), label="ratio")
#plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
#plt.xlim((-20, 20))
plt.ylim((10**(-1), 10**(2)))
plt.title("FFT ratio of beam images")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend()
plt.show()

fft_spydG[:middle-R_1, :] = spyd_low
fft_spydG[middle-R_1:, :] = spyd_high
fft_back_spyd = abs(np.fft.ifft2(fft_spydG))

# Plotting the back transformation
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(abs(fft_spydG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

        
plt.subplot(212)
plt.imshow(fft_back_spyd, origin='lower', aspect=aspect_rad, vmin=0, vmax=0.25, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()

plt.tight_layout()
#plt.savefig("interpolation/HDwarped_R290_R490.pdf")
plt.show()




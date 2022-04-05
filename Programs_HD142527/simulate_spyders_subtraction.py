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
from transformations_functions import to_rphi_plane
from scipy.optimize import curve_fit
from filter_functions import gaussianBeam, gaussianSpyder, Gaussian1D, e_func
import aotools
from aperture_fluxes import aperture_flux_image, aperture_flux_warped


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

# Warp the image into the r-phi plane
warp = to_rphi_plane(zeros, (x_len, y_len), R_1, R_2)
warp_or = warp.T
warp_shape = warp.shape

aspect_value = (360/warp_shape[0])/((R_2-R_1)/warp_shape[1])
aspect_rad = (2*np.pi/warp_shape[0])/((R_2-R_1)/warp_shape[1])


radi = np.arange(warp_shape[1])
phis = np.arange(warp_shape[0])/warp_shape[0]*2*np.pi
cen_r = int((R_2-R_1)/2)

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
spos = [42,  670] 
spos_shift = [42+shift,  670+shift]   #670+shift
degsym = 180/360*warp_shape[0]


# We insert smoothed (gaussian) beams at the positions of the spiders with 
# the same width and intensity
beamG1 = gaussianBeam(warp_or.copy(), spos_shift[0], 5)*1.7
beamG2 = gaussianBeam(warp_or.copy(), spos_shift[1], 10)*2.5
beamG3 = gaussianBeam(warp_or.copy(), spos_shift[0]+degsym, 8)*2.1
beamG4 = gaussianBeam(warp_or.copy(), spos_shift[1]+degsym, 6)*0.8

#gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, I_g)
#gauss_inner = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_gi, I_g)

beamG_shift = beamG1 + beamG2 + beamG3 + beamG4
fft_beamG_shift = np.fft.fftshift(np.fft.fft2(beamG_shift))

# We want to plot  the fft
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 50*aspect_value))  
ax1[0].plot(phis, beamG_shift[cen_r, :], 
            label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
            %(5/warp_shape[0]*2*np.pi, 10/warp_shape[0]*2*np.pi, 8/warp_shape[0]*2*np.pi, 
              6/warp_shape[0]*2*np.pi))
ax1[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax1[0].set_xlabel(r'$\varphi$ [rad]')
ax1[0].legend(loc='upper right')

ax1[1].plot(phi_freq, abs(fft_beamG_shift[cen_r, :]), label="FFT")
ax1[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
#ax1[1].set_ylim((10**(-3), 4*10**(4)))
ax1[1].set_xlim((-20, 20))
ax1[1].legend()
plt.savefig("fourier/Gaussian_spider_simulation.pdf")
plt.show()


###############################################################################
###############################################################################
###############################################################################

# We insert smoothed (gaussian) simulated spyders at the positions of the spyders        
spydG1 = gaussianSpyder(warp_or.copy(), spos[0], 5, 1.7)
spydG2 = gaussianSpyder(warp_or.copy(), spos[1], 10, 2.5)
spydG3 = gaussianSpyder(warp_or.copy(), spos[0]+degsym, 8, 2.1)
spydG4 = gaussianSpyder(warp_or.copy(), spos[1]+degsym, 6, 0.8)

spydG = spydG1 + spydG2 + spydG3 + spydG4
  
# Fourier transform the warped image with beams
fft_spydG = np.fft.fftshift(np.fft.fft2(spydG))

# Define an other set of spiders, but these are shifted by a certain amount
# We do this to find out how the position influences the FFT
spydG1_shift = gaussianSpyder(warp_or.copy(), spos_shift[0], 5, 1.7)
spydG2_shift = gaussianSpyder(warp_or.copy(), spos_shift[1], 10, 2.5)
spydG3_shift = gaussianSpyder(warp_or.copy(), spos_shift[0]+degsym, 8, 2.1)
spydG4_shift = gaussianSpyder(warp_or.copy(), spos_shift[1]+degsym, 6, 0.8)

spydG_shift = spydG1_shift + spydG2_shift + spydG3_shift + spydG4_shift
fft_spydG_shift = np.fft.fftshift(np.fft.fft2(spydG_shift))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(spydG, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
#plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
 #                                                 r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_spydG + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], 
                                       radi_freq[-1]])
#plt.imshow(fft_spydG.imag, origin='lower', cmap='gray', aspect=aspect_freq, vmin=-100, 
#           vmax= 100, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
plt.savefig("fourier/simulated_spyder.pdf")
plt.show()

"""
y = 0            
plt.figure(figsize=(8, 16*aspect_value))
while y < (R_2-R_1): 
    plt.plot(phis, spydG[y, :], label="at y = %.0f" %(y+R_1))
    y += 50
plt.title("Horizontal cut through the spiders")
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.legend()
plt.show()
"""

# We define the parameters for different Gaussian which can potentially be used
# for the subtraction
w_g = 68
w_gi = 55
I_g = 9.918*10**3
I_gm = 9.9*10**3
gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, I_g)
gauss_inner = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_gi, I_g)

w = 61
neg_r = 40 
ratio_gauss = np.sum(abs(fft_spydG[cen_r+1:cen_r+neg_r, int(len(phis)/2)-w:
                                 int(len(phis)/2)-1]), axis=1)/np.sum(abs(
                                     fft_spydG[cen_r, int(len(phis)/2)-w:
                                               int(len(phis)/2)-1]))
print(ratio_gauss)
w_s = 40
gauss_s = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_s, 0.84*10**4)*ratio_gauss[0]   #Neg_r verändern evtl dann besser
                                         
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydG[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss), label="Gaussian $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
#plt.semilogy(phi_freq, abs(gauss_inner + 0.0001), label="Gaussian profile")
#plt.semilogy(phi_freq, abs(gauss_s + 0.0001), label="Gaussian profile")
plt.ylim((10**(-3), 2*10**(4)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y == int((R_2-R_1)/2):
    plt.plot(phi_freq, fft_spydG[y, :].real, label ="radial freq. = %.2f" %(radi_freq[y]))
    plt.plot(phi_freq, fft_spydG_shift[y, :].real, label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.plot(phi_freq, abs(gauss), label="Gaussian $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
#plt.plot(phi_freq, abs(gauss_inner + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_gi))
#plt.plot(phi_freq, abs(gauss_s + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_s))
plt.xlim((-60, 60))
#plt.ylim((10**(-3), 10**(4)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_angularfreq_enlarged.pdf")
plt.show()

# We subtract the fft of the spiders from the one of the shifted spider to 

y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
plt.plot(phi_freq, fft_spydG[y, :].real-fft_spydG_shift[y, :].real, label ="radial freq. = %.2f" %(radi_freq[y]))
#plt.ylim((10**(-3), 2*10**(4)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_angularfreq_enlarged.pdf")
plt.show()


# We compare the fft of the simulated spider and the simulated gaussian beam at
# central radial frequency
plt.figure(figsize=(8, 24*aspect_value))
plt.semilogy(phi_freq, abs(fft_spydG[cen_r, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
plt.semilogy(phi_freq, abs(fft_beamG_shift[cen_r, :]), label="Gaussian beams")
plt.ylim((10**(-3), 4*10**(4)))
plt.xlim((-70, 70))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_angularfreq.pdf")
plt.show()

"""
######################### Add PSF #############################################
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
warp_psf = warp_psf.T

spydPSF = spydG.copy() + warp_psf

# Fourier transform the warped image with beams
fft_spydPSF = np.fft.fftshift(np.fft.fft2(spydPSF))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(spydPSF, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
#plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
 #                                                 r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_spydPSF + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], 
                                       radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
plt.savefig("fourier/simulated_spyder_PSF.pdf")
plt.show()

y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydPSF[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
    #plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss), label="Gaussian $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
plt.ylim((10**(-3), 2*10**(4)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_PSF_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydPSF[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss), label="Gaussian $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
plt.xlim((-60, 60))
plt.ylim((10**(-3), 2*10**(4)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_angularfreq_enlarged.pdf")
plt.show()

y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydPSF[y, :] - fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    #plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
#plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian profile")
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydPSF[y, :] - fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
#plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian profile")
plt.xlim((-60, 60))
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_angularfreq_enlarged.pdf")
plt.show()


x = int(len(phis)/2)
plt.figure(figsize=(8, 5))
while x > 0:
    plt.semilogy(radi_freq, abs(fft_spydPSF[:, x] + 0.0001), label ="phi pos. = %.i" %(phi_freq[x]))
    if abs(phi_freq[x]) > 50:
        x -= 500
    else:
        x -= 80
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_radialfreq.pdf")
plt.show()

######################### Add noise ###########################################

spydN = spydG.copy() + np.random.rand(warp_shape[1], warp_shape[0])/4*3

# Fourier transform the warped image with beams
fft_spydN = np.fft.fftshift(np.fft.fft2(spydN))
        
# Plotting the warped and fft of it
plt.figure(figsize=(8, 16*aspect_value))
        
plt.subplot(211)
plt.imshow(spydN, origin='lower', aspect=aspect_rad, vmin=0, vmax=Imax, 
           extent=[0, 2*np.pi, R_1, R_2])
plt.xlabel(r'$\varphi$ [rad]')
plt.ylabel('Radius')
#plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
 #                                                 r'$3\pi/2$', r'$2\pi$'])
plt.colorbar()
        
plt.subplot(212)
plt.imshow(abs(fft_spydN + 0.0001), origin='lower', cmap='gray',  norm=LogNorm(vmin=1),
           aspect=aspect_freq, extent=[phi_freq[0], phi_freq[-1], radi_freq[0], 
                                       radi_freq[-1]])
plt.xlabel(r'Frequency [$\frac{1}{\mathrm{rad}}$]')
plt.ylabel(r'Frequency [$\frac{1}{\mathrm{px}}$]')
plt.ylim((-0.5, 0.5))
plt.colorbar()

plt.tight_layout()
plt.savefig("fourier/simulated_spider_noise.pdf")
plt.show()

y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydN[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
    #plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss), label="Gaussian $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
plt.semilogy(phi_freq, abs(gauss_inner), label="Gaussian $\sigma$ = %.1f" %(w_gi/warp_shape[0]*phi_freq[-1]*2))
plt.ylim((10**(-2), 4*10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_noise_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydN[y, :]), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss), label="Gaussian $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
plt.semilogy(phi_freq, abs(gauss_inner), label="Gaussian $\sigma$ = %.1f" %(w_gi/warp_shape[0]*phi_freq[-1]*2))
plt.semilogy(phi_freq, abs(gauss_s), label="Gaussian $\sigma$ = %.1f" %(w_s/warp_shape[0]*phi_freq[-1]*2))
plt.xlim((-60, 60))
plt.ylim((10**(-2), 4*10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_angularfreq_enlarged.pdf")
plt.show()

y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydN[y, :] - fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    #plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
#plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian profile")
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_angularfreq.pdf")
plt.show()


x = int(len(phis)/2)
plt.figure(figsize=(8, 5))
while x > 0:
    plt.semilogy(radi_freq, abs(fft_spydN[:, x] + 0.0001), label ="phi pos. = %.i" %(phi_freq[x]))
    if abs(phi_freq[x]) > 50:
        x -= 500
    else:
        x -= 80
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.savefig("fourier/simspi_radialfreq.pdf")
plt.show()
"""

########## The image path of the images taken in the P2 mode #################
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

# We define the positions of the ghosts
gh_pos = [(891.0, 599.0), (213.0, 387.0)]

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
        
        spos = [42,  670]
        
        # Choose the intensity
        Imax = 5
        Imax_small = 0.5
        Imin_small = -0.5
        
        # PSF
        zeros = np.zeros((x_len, y_len))
        mask_psf = aotools.circle(64,128)-aotools.circle(16, 128)
        zeros[:128,:128] = mask_psf 
        psf = aotools.ft2(zeros, delta=1./128.,)
        psf = abs(psf)/1000
        
        psf_pos = (300, 800)
        psf[700:900, 200:400] = psf[512-100:512+100, 512-100:512+100]
        psf[512-100:512+100, 512-100:512+100] = psf[0:200, 0:200]
        
        int1 = int1 + psf
        
        ## Computation of the aperture flux of the ghost
        model_planet = psf_pos  #gh_pos[1]
        f_ap_im, ap_im, annu_im = aperture_flux_image(int1, model_planet)
        print("The aperture flux of the model planet in the original image is: ", f_ap_im)
        

        # Warp the image to the r-phi Plane    
        warped = to_rphi_plane(int1, (x_len, y_len), R_1, R_2)
        warped_shape = warped.shape
        warped = warped.T
        
        # We take out the intensity change in radial direction due to the
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
        """
        # We create a Gaussian profile
        beamG = Gaussian1D(phis.copy(), int(2.4*degsym/4), 8)*2.7


        plt.figure(figsize=(8, 21*aspect_value))
        plt.plot(phis, warped[1, :], label="Image intensity at radius 255")
        #plt.title("Horizontal cut through the beam images")
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
                   [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 21*aspect_value))
        plt.plot(phis[int(degsym/2):int(3/4*degsym)], warped[1, int(degsym/2):int(3/4*degsym)], label="Image intensity at radius 255")
        plt.plot(phis[int(degsym/2):int(3/4*degsym)], beamG[int(degsym/2):int(3/4*degsym)], label="Gaussian fit")
        #plt.title("Horizontal cut through the beam images")
        plt.xticks([np.pi/2, 3/4*np.pi], [r'$\pi/2$', r'$3\pi/4$'])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.legend()
        plt.tight_layout()
        plt.savefig("fourier/spyder_gaussian.pdf")
        plt.show()
        """
        
        ## Plot the output and its fft
        fourier = np.fft.fftshift(np.fft.fft2(warped))
        fourier_real = fourier.real
      
        plt.figure(figsize=(8, 16*aspect_value))

        plt.subplot(211)
        plt.imshow(warped, origin='lower', aspect=aspect_rad, vmin=Imin_small, 
                   vmax= Imax_small, extent=[0, 2*np.pi, R_1, R_2])
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
        #plt.savefig("suppression/HDflatten_R254_R454_-1to1.pdf")
        plt.show()
        
        ################ Spyder suppression central freq ######################
        # Now the final subtraction = division by the gaussian
        I_g = 9.918*10**3
        gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, I_g)

        suppr = fft_spydG[cen_r, :]
        
        #Subtract it
        spid_center = fourier[cen_r, :].copy()
        fourier[cen_r, :] = spid_center[:]-suppr[:]
     
        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spid_center), label="spid")
        plt.semilogy(phi_freq, abs(suppr)+0.001, label="gauss $\sigma$ = %.1f" %(w_g/warp_shape[0]*phi_freq[-1]*2))
        plt.xlim((-50, 50))
        plt.ylim((10**(-1), 10**(5)))
        plt.title("FFT ratio of beam images")
        plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
        plt.legend()
        plt.show()
    
        warped_back = np.fft.ifft2(np.fft.ifftshift(fourier)).real
        
        ## Computation of the aperture flux of the model planet in the flattened 
        ## and FFT back where a gaussian is subtracted from the center
        f_ap_fft, ap_fft_draw, annu_fft_draw = aperture_flux_warped(warped_back, 
                                                                    warped_shape, R_1, 
                                                                    aspect_rad, 
                                                                    model_planet)
        print("The aperture flux of the model planet without (only at radial freq=0) spyders is: ", f_ap_fft)
        print("This corresponds to ", round(100/f_ap_f*f_ap_fft, 3), " %")
        
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
        #plt.savefig("suppression/HDcentralfreq_R254_R454_-1to1.pdf")
        plt.show()
        """
        
        ######################################################################
        ######### Frequency suppression also in radial direction #############
        ######################################################################
        
        # Ratio in radial direction in order to make a Gaussian subtraction of all 
        # frequencies in radial direction (also the larger ones)
        r_n = cen_r - 1
        r_p = cen_r + 1
        ratio_i = 0
        while r_n > cen_r - 3:
            w_s = int(0.84*w * ratio_gauss[ratio_i])
            suppr_small_n = fft_spydG[r_n, :]
            suppr_small_p = fft_spydG[r_p, :]
            fourier[r_n, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s] = fourier[
                r_n, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s]/(
                   suppr_small_n[int(len(phis)/2)-w_s:int(len(phis)/2)+w_s])
            fourier[r_p, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s] = fourier[
                r_p, int(len(phis)/2)-w_s:int(len(phis)/2)+w_s]/(
                    suppr_small_p[int(len(phis)/2)-w_s:int(len(phis)/2)+w_s])
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
        """

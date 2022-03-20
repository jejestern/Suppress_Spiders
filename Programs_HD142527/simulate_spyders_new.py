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
print(warp_shape[0])

In = [1, 5]

fig1, ax = plt.subplots(2, 1, figsize=(8, 40*aspect_value))    
for i in range(4):
    deg = 0
    if i >= 2:
        i -= 2
        deg = degsym
        
    img_G = gaussianBeam(warp_or.copy(), spos[i] + deg, 12)*In[i]
    
    ax[0].plot(phis, img_G[middle-R_1, :], label="Gaussian beams")
    
    # Fourier transform the warped image with beams
    fft_G = np.fft.fftshift(np.fft.fft2(img_G))
    
    ax[1].semilogy(phi_freq, abs(fft_G[middle-R_1, :] + 0.0001), label="Gaussian beams")
    
    
ax[0].set_title("Horizontal cut through")
ax[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax[0].legend()

#ax2.set_ylim((10**(-1), 10**(5)))
ax[1].set_title("FFT of beam images")
ax[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
ax[1].legend()

plt.show()
    
fig1, ax = plt.subplots(2, 1, figsize=(8, 45*aspect_value))  
width_G = [5, 10, 15, 20] 

for i in width_G:
        
    img_G = gaussianBeam(warp_or.copy(), spos[1], i)
    
    ax[0].plot(phis[int(degsym/2):int(3/4*degsym)], 
               img_G[middle-R_1, int(degsym/2):int(3/4*degsym)], label=r"$\sigma$ = %.3f" %(i/warp_shape[0]*2*np.pi))
    
    # Fourier transform the warped image with beams
    fft_G = np.fft.fftshift(np.fft.fft2(img_G))
    
    ax[1].semilogy(phi_freq, abs(fft_G[middle-R_1, :] + 0.0001), label=r"$\sigma$ = %.3f" %(i/warp_shape[0]*2*np.pi))
    
    
ax[0].set_title("Gaussian profiles")
ax[0].set_xticks([np.pi/2, 3/4*np.pi], [r'$\pi/2$', r'$3\pi/4$'])
ax[0].set_xlabel(r'$\varphi$ [rad]')
ax[0].legend()

#ax2.set_ylim((10**(-1), 10**(5)))
ax[1].set_title("FFT")
ax[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
ax[1].legend()

plt.tight_layout()
plt.savefig("fourier/Gauss_diffwidths.pdf")
plt.show()


# We insert smoothed (gaussian) beams at the positions of the spyders
beamG1 = gaussianBeam(warp_or.copy(), spos[0], 12)*1.7
beamG2 = gaussianBeam(warp_or.copy(), spos[1], 12)*2.5
beamG3 = gaussianBeam(warp_or.copy(), spos[0]+degsym, 12)*2.1
beamG4 = gaussianBeam(warp_or.copy(), spos[1]+degsym, 12)*0.8

beamG = beamG1 + beamG2 + beamG3 + beamG4
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))
fft_G1 = np.fft.fftshift(np.fft.fft2(beamG1))
fft_G2 = np.fft.fftshift(np.fft.fft2(beamG2))
fft_G3 = np.fft.fftshift(np.fft.fft2(beamG3))
fft_G4 = np.fft.fftshift(np.fft.fft2(beamG4))

# We want to plot  the mean of the fft
fft_mean = fft_beamG[middle-R_1, :].copy()
for i in range(len(fft_mean)):
    if i > 100 and i < len(fft_mean) - 100:
        fft_mean[i] = np.mean(fft_mean[i-4:i+4])

fig2, ax2 = plt.subplots(2, 1, figsize=(8, 48*aspect_value))  
ax2[0].plot(phis, beamG[middle-R_1, :], label="Gaussian spiders $\sigma$ = %.3f" %(12/warp_shape[0]*2*np.pi))
ax2[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax2[0].set_xlabel(r'$\varphi$ [rad]')
ax2[0].legend(loc='upper right')
        
ax2[1].semilogy(phi_freq[900:-900], abs(fft_beamG[middle-R_1, 900:-900] + 0.0001), label="FFT")
#ax2[1].semilogy(phi_freq[900:-900], abs(fft_mean[900:-900] + 0.0001), label="Averaged FFT")
ax2[1].semilogy(phi_freq[900:-900], abs(fft_G1[middle-R_1, 900:-900] + 0.0001), label="FFT of Gaussian profile 1")
ax2[1].semilogy(phi_freq[900:-900], abs(fft_G2[middle-R_1, 900:-900] + 0.0001), label="FFT of Gaussian profile 2")
ax2[1].semilogy(phi_freq[900:-900], abs(fft_G3[middle-R_1, 900:-900] + 0.0001), label="FFT of Gaussian profile 3")
ax2[1].semilogy(phi_freq[900:-900], abs(fft_G4[middle-R_1, 900:-900] + 0.0001), label="FFT of Gaussian profile 4")
ax2[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
ax2[1].legend()

plt.tight_layout()
plt.savefig("fourier/Gaussian_fourheightspyders.pdf")
plt.show()
"""
fft_back= abs(np.fft.ifft(fft_G1[middle-R_1, :]))
plt.figure()
plt.plot(phis, fft_back)
plt.show()
"""

# We insert smoothed (gaussian) beams at the positions of the spyders
beamG1 = gaussianBeam(warp_or.copy(), spos[0], 10)
beamG2 = gaussianBeam(warp_or.copy(), spos[1], 15)
beamG3 = gaussianBeam(warp_or.copy(), spos[0]+degsym, 20)
beamG4 = gaussianBeam(warp_or.copy(), spos[1]+degsym, 5)

beamG = beamG1 + beamG2 + beamG3 + beamG4
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))

# We want to plot  the mean of the fft
fft_mean = fft_beamG[middle-R_1, :].copy()
for i in range(len(fft_mean)):
    if i > 100 and i < len(fft_mean) - 100:
        fft_mean[i] = np.mean(fft_beamG[middle-R_1, i-4:i+4])

fig3, ax3 = plt.subplots(2, 1, figsize=(8, 50*aspect_value))  
ax3[0].plot(phis, beamG[middle-R_1, :], 
            label=r"Gaussian spiders: $\sigma_1$ = %.3f, $\sigma_2$ = %.3f, $\sigma_3$ = %.3f, $\sigma_4$ = %.3f" 
            %(10/warp_shape[0]*2*np.pi, 15/warp_shape[0]*2*np.pi, 20/warp_shape[0]*2*np.pi, 
              5/warp_shape[0]*2*np.pi))
ax3[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax3[0].set_xlabel(r'$\varphi$ [rad]')
ax3[0].legend(loc='upper right')

ax3[1].semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="FFT")
#ax3[1].semilogy(phi_freq, abs(fft_mean + 0.0001), label="Averaged FFT")
ax3[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
ax3[1].legend()
plt.savefig("fourier/Gaussian_fourdiffspyders.pdf")
plt.show()
"""
fft_back= abs(np.fft.ifft(fft_mean))
plt.figure()
plt.plot(phis, fft_back)
plt.show()
"""

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

w_g = 68
w_gi = 55
I_g = 9.918*10**3
I_gm = 9.9*10**3
gauss = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_g, I_g)
gauss_inner = Gaussian1D(phi_freq.copy(), int(len(phis)/2), w_gi, I_g)

cen_r = int((R_2-R_1)/2)
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
    plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_g))
plt.semilogy(phi_freq, abs(gauss_inner + 0.0001), label="Gaussian profile")
plt.semilogy(phi_freq, abs(gauss_s + 0.0001), label="Gaussian profile")
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.plot(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.plot(phi_freq, abs(gauss + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_g))
plt.plot(phi_freq, abs(gauss_inner + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_gi))
plt.plot(phi_freq, abs(gauss_s + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_s))
plt.xlim((-60, 60))
#plt.ylim((10**(-4), 2*10**(4)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_angularfreq_enlarged.pdf")
plt.show()


x = int(len(phis)/2)
plt.figure(figsize=(8, 5))
while x > 0:
    plt.semilogy(radi_freq, abs(fft_spydG[:, x] + 0.0001), label ="phi pos. = %.i" %(phi_freq[x]))
    if abs(phi_freq[x]) > 50:
        x -= 500
    else:
        x -= 80
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_radialfreq.pdf")
plt.show()

## Fitting an exponential to the horizontal through the center frequency
popt, pcov = curve_fit(oneover_x, radi_freq[middle-R_1+1:] , abs(fft_spydG[middle-R_1+1:, int(len(phis)/2)]))

plt.figure()
plt.semilogy(radi_freq[middle-R_1+1:], abs(fft_spydG[middle-R_1+1:, int(len(phis)/2)]), '.', label="phi pos. = 0")
plt.semilogy(radi_freq[middle-R_1+1:], oneover_x(radi_freq[middle-R_1+1:], *popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images horizontal")
plt.xlabel(r'Radial frequency [$\frac{1}{\mathrm{px}}$]')
plt.legend(loc='upper right')
plt.show()

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
    plt.semilogy(phi_freq, abs(fft_spydPSF[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    #plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_g))
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_PSF_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.plot(phi_freq, abs(fft_spydPSF[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.plot(phi_freq, abs(gauss + 0.0001), label="Gaussian profile")
plt.xlim((-60, 60))
#plt.ylim((10**(-1), 10**(5)))
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
    plt.semilogy(phi_freq, abs(fft_spydN[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    #plt.semilogy(phi_freq, abs(fft_spydG[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_g))
plt.semilogy(phi_freq, abs(gauss_inner + 0.0001), label="Gaussian $\sigma$ = %.i" %(w_gi))
#plt.ylim((10**(-1), 10**(5)))
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("fourier/simspi_noise_angularfreq.pdf")
plt.show()

# The same as before, but with an enlarged x-range
y = int((R_2-R_1)/2)
plt.figure(figsize=(8, 24*aspect_value))
while y > 0:
    plt.semilogy(phi_freq, abs(fft_spydN[y, :] + 0.0001), label ="radial freq. = %.2f" %(radi_freq[y]))
    if y == int((R_2-R_1)/2):
        y -= 1
    elif abs(radi_freq[y]) < 0.04:
        y -= 5
    else:
        y -= 40
plt.semilogy(phi_freq, abs(gauss + 0.0001), label="Gaussian profile")
plt.semilogy(phi_freq, abs(gauss_inner + 0.0001), label="Gaussian profile")
plt.semilogy(phi_freq, abs(gauss_s + 0.0001), label="Gaussian profile")
plt.xlim((-60, 60))
plt.ylim((10**(-1), 10**(5)))
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
        w = 61
        spid_center = fourier[middle-R_1, :].copy()
        fourier[middle-R_1, int(len(phis)/2)-w:int(len(phis)/2)+w] = spid_center[int(len(phis)/2)-
                                                                         w:int(len(phis)/2)
                                                                         +w]/suppr[int(len(phis)/2)-w:int(len(phis)/2)+w]
     
        plt.figure(figsize=(8, 16*aspect_value))
        plt.semilogy(phi_freq, abs(spid_center), label="spid")
        plt.semilogy(phi_freq, abs(suppr)+0.001, label="gauss $\sigma$ = %.i" %(w_g))
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
        

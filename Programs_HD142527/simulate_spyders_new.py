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
#degn = 90/360*warp_shape[0]
spos = [42+shift,  670+shift]   
degsym = 180/360*warp_shape[0]
print(warp_shape[0])
fig1, ax = plt.subplots(2, 1, figsize=(8, 40*aspect_value))    
for i in range(4):
    deg = 0
    if i >= 2:
        i -= 2
        deg = degsym
        
    img_G = gaussianBeam(warp_or.copy(), spos[i] + deg, 12)
    
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
beamG1 = gaussianBeam(warp_or.copy(), spos[0], 12)
beamG2 = gaussianBeam(warp_or.copy(), spos[1], 12)
beamG3 = gaussianBeam(warp_or.copy(), spos[0]+degsym, 12)
beamG4 = gaussianBeam(warp_or.copy(), spos[1]+degsym, 12)

beamG = beamG1 + beamG2 + beamG3 + beamG4
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))
fft_G1 = np.fft.fftshift(np.fft.fft2(beamG1))

# We want to plot  the mean of the fft
fft_mean = fft_beamG[middle-R_1, :].copy()
for i in range(len(fft_mean)):
    if i > 100 and i < len(fft_mean) - 100:
        fft_mean[i] = np.mean(fft_mean[i-4:i+4])

fig2, ax2 = plt.subplots(2, 1, figsize=(8, 48*aspect_value))  
ax2[0].plot(phis, beamG[middle-R_1, :], label="Gaussian spyders")
ax2[0].set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
ax2[0].set_xlabel(r'$\varphi$ [rad]')
ax2[0].legend(loc='upper right')
        
ax2[1].semilogy(phi_freq[900:-900], abs(fft_beamG[middle-R_1, 900:-900] + 0.0001), label="FFT")
#ax2[1].semilogy(phi_freq[900:-900], abs(fft_mean[900:-900] + 0.0001), label="Averaged FFT")
ax2[1].semilogy(phi_freq[900:-900], abs(fft_G1[middle-R_1, 900:-900] + 0.0001), 'tab:green', label="FFT of Gaussian profile")
#plt.ylim((10**(-1), 10**(5)))
ax2[1].set_xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
ax2[1].legend()

plt.tight_layout()
plt.savefig("fourier/Gaussian_fourspyders.pdf")
plt.show()
"""
fft_back= abs(np.fft.ifft(fft_G1[middle-R_1, :]))
plt.figure()
plt.plot(phis, fft_back)
plt.show()
"""

# We insert smoothed (gaussian) beams at the positions of the spyders
beamG1 = gaussianBeam(warp_or.copy(), spos[0], 10)
beamG2 = gaussianBeam(warp_or.copy(), spos[1], 10)
beamG3 = gaussianBeam(warp_or.copy(), spos[0]+degsym, 20)
beamG4 = gaussianBeam(warp_or.copy(), spos[1]+degsym, 12)

beamG = beamG2 + beamG3
  
# Fourier transform the warped image with beams
fft_beamG = np.fft.fftshift(np.fft.fft2(beamG))

plt.figure(figsize=(8, 16*aspect_value))
plt.plot(phis, beamG[middle-R_1, :], label="Gaussian beams")
plt.title("Horizontal cut through the beam images")
plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                  r'$3\pi/2$', r'$2\pi$'])
plt.legend()
plt.show()

plt.figure(figsize=(8, 16*aspect_value))
plt.semilogy(phi_freq, abs(fft_beamG[middle-R_1, :] + 0.0001), label="Gaussian beams")
#plt.ylim((10**(-1), 10**(5)))
plt.title("FFT of beam images")
plt.xlabel(r'Angular frequency [$\frac{1}{\mathrm{rad}}$]')
plt.legend()
plt.show()
"""
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

"""
########## The image path of the images taken in the P2 mode ############
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

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
        
        # We create a Gaussian profile
        beamG1 = gaussianBeam(warped.copy(), spos[0], 10)

        beamG = Gaussian1D(phis.copy(), int(2.4*degsym/4), 8)*2.7


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
        
        
        # Plotting the back transformation
        plt.figure(figsize=(8, 10*aspect_value))

        plt.imshow(warped, origin='lower', aspect=aspect_rad, vmin=0, vmax=2, 
                   extent=[0, 2*np.pi, R_1, R_2])
        plt.xlabel(r'$\varphi$ [rad]')
        plt.ylabel('Radius')
        plt.xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$\pi/2$', r'$\pi$', 
                                                          r'$3\pi/2$', r'$2\pi$'])
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("fourier/warped_254_454.pdf")
        plt.show()


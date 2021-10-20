"""
This program finds the radius needed for the aperture photometry.

2021-10-06
Jennifer Studer <studerje@student.ethz.ch>
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sys import argv, exit
import os
from photutils import CircularAperture, aperture_photometry, find_peaks
from astropy.visualization import simple_norm
from astropy.table import Table

# This part takes the argument and saves the folder 
if not len(argv) == 1:
    print("Wrong number of arguments!")
    print("Usage: python find_radius.py")
    print("Exiting...")
    exit()

path = "/home/jeje/Dokumente/Masterthesis/ZirkumstellareScheibe_HD142527"
files = os.listdir(path)


for image_name in files[0:1]:
    if image_name.endswith(".fits"):
        img_data = fits.getdata(path + "/" + image_name, ext=0)
        fits.info(path + "/" + image_name)
        data = img_data[0,:,:]

        tbl = find_peaks(data, 300, box_size=50)
        pe = np.array(tbl['peak_value'])
        pe_x = np.array(tbl['x_peak'])
        pe_y = np.array(tbl['y_peak'])
        peaks = np.array((pe_x, pe_y, pe)).T
        peaks = peaks.tolist()
        peaks = sorted(peaks, key=lambda t: t[2], reverse=True)
        positions = (peaks[0][0], peaks[0][1])
        
        apertures = lambda r: CircularAperture(positions, r=r)
        phot_table = lambda r: aperture_photometry(data, apertures(r))

        rad = [0.0]
        numcounts = [0.0]
        noise = 10000
        i = 0
        while(noise >= 5000):
            rad.append(rad[i] + 1.0)
            numcounts.append(phot_table(rad[i+1])['aperture_sum'])
            noise = numcounts[i+1] - numcounts[i]
            i += 1

        rad = np.array(rad)
        numcounts = np.array(numcounts)
        radius = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        aper = np.zeros_like(radius)
        for ap in range(len(radius)):
            aper[ap] = phot_table(radius[ap])['aperture_sum']

        plt.plot(radius, aper, 'o', label="The counts against a certain radius")
        plt.plot(rad, numcounts, 'rx', label="Choosen radius")
        plt.xlabel('Radius r in pixels')
        plt.ylabel('Counts')
        plt.legend()
        #plt.title(star) 
        plt.show()
        print('The radius to choose is r = ', rad[-1])


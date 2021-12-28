"""
This program examinates the ghosts of the circumstellar disk. It uses aperture 
photometry to do so and the images used are in P2, meaning the ghosts stay at 
the same position in all images. 

Created on 2021-10-13
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sys import argv, exit
import os
from photutils import find_peaks
from astropy.visualization import simple_norm
from aperture_radius import aperture_phot


# This part takes the argument and saves the folder 
if not len(argv) == 1:
    print("Wrong number of arguments!")
    print("Usage: python ghosts.py")
    print("Exiting...")
    exit()

# The image path of the images taken in the P2 mode
path = "/home/jeje/Dokumente/Masterthesis/Programs_HD142527/ZirkumstellareScheibe_HD142527/P2_mode"
files = os.listdir(path)

# Initialisation of lists in which we save the information
aperture_rad = 6.0
image_names = []
SN_1 = []
SN_2 = []
fmean_1 = []
fmean_2 = []
sigma_1 = []
sigma_2 = []
ratio_1 = []
ratio_2 = []

# We define the positions of the ghosts
gh_pos = [(891.0, 600.0), (213.0, 387.0)]

for image_name in files:
    if image_name.endswith("1.fits"): 
        # Reading in the images from camera 1
        img_data = fits.getdata(path + "/" + image_name, ext=0)
        fits.info(path + "/" + image_name)
        image_names.append(image_name[4:10])

        # Vertical flip image data to have the same convention as ds9
        #axis2fl=int(img_data.ndim-2)
        #print('axis to flip:',axis2fl)
        #img_ori = np.flip(img_data, axis2fl)

        # Choose the intensity 1
        int1 = img_data[0,:,:]
        
        # Find the position of the star
        tbl = find_peaks(int1, 500, box_size= 50)
        pe = np.array(tbl['peak_value'])
        pe_x = np.array(tbl['x_peak'])
        pe_y = np.array(tbl['y_peak'])
        peaks = np.array((pe_x, pe_y, pe)).T
        peaks = peaks.tolist()
        peaks = sorted(peaks, key=lambda t: t[2], reverse=True)
        star_position = [(peaks[0][0], peaks[0][1])]
        
        # Define the norm for afterwards plotting the image
        norm = simple_norm(int1, 'log', percent=99.9)
        
        # We loop over the different ghosts
        for i in range(len(gh_pos)):
            fmean, sigma, SN, ratio = aperture_phot(int1, norm, gh_pos[i], star_position, aperture_rad, True)
            
            if i == 0:
                SN_1.append(SN)
                fmean_1.append(fmean)
                sigma_1.append(sigma)
                ratio_1.append(ratio)
            else:
                SN_2.append(SN)
                fmean_2.append(fmean)
                sigma_2.append(sigma)
                ratio_2.append(ratio)

            
        # Plot the image
        fig1, ax = plt.subplots(figsize=(12,12))
        im = ax.imshow(int1, cmap='gray', norm = norm)

        # Creation of the plot, with the zoomed in stuff
        ax.set(xlim=(0, int1.shape[1]-1), ylim=(0, int1.shape[0]-1))
        ## inset axes for zoom
        axins1 = ax.inset_axes([-0.8, 0.55, 0.6, 0.6])
        axins1.imshow(int1, cmap='gray', norm = norm)
        axins2 = ax.inset_axes([-0.8, -0.1, 0.6, 0.6])
        axins2.imshow(int1, cmap='gray', norm = norm)
        ## sub region of the original image 
        axins1.set_xlim(gh_pos[0][0]-30, gh_pos[0][0]+30)
        axins1.set_ylim(gh_pos[0][1]-30, gh_pos[0][1]+30)
        axins1.set_xticklabels('')
        axins1.set_yticklabels('')
        axins2.set_xlim(gh_pos[1][0]-30, gh_pos[1][0]+30)
        axins2.set_ylim(gh_pos[1][1]-30, gh_pos[1][1]+30)
        axins2.set_xticklabels('')
        axins2.set_yticklabels('')

        ax.indicate_inset_zoom(axins1, edgecolor="yellow")
        ax.indicate_inset_zoom(axins2, edgecolor="yellow")

        fig1.colorbar(im)
        fig1.tight_layout()
        plt.show()
        
            
# Calculation of the means
SN_1_mean = np.mean(SN_1)
SN_2_mean = np.mean(SN_2)
ratio_1_mean = np.mean(ratio_1)
ratio_2_mean = np.mean(ratio_2)
fmean_1_mean = np.mean(fmean_1)
fmean_2_mean = np.mean(fmean_2)
sigma_1_mean = np.mean(sigma_1)
sigma_2_mean = np.mean(sigma_2)

ratio_1_mean_err = ratio_1_mean * SN_1_mean/100
ratio_2_mean_err = ratio_2_mean * SN_2_mean/100

# Write the data in a txt file
ghosts = open("/home/jeje/Dokumente/Masterthesis/Programs_HD142527/aperture_photometry/ghosts.txt", "w")
ghosts.write(str('image_names, SN of ghost 1, SN of ghost 2, ratio 1, ratio 2, fmean 1, fmean 2, sigma 1, sigma 2')+'\n')
ghosts.write(str(image_names)+'\n') 
ghosts.write(str(SN_1)+'\n')   
ghosts.write(str(SN_2)+'\n')
ghosts.write(str(ratio_1)+'\n')   
ghosts.write(str(ratio_2)+'\n')
ghosts.write(str(fmean_1)+'\n')   
ghosts.write(str(fmean_2)+'\n')
ghosts.write(str(sigma_1)+'\n')   
ghosts.write(str(sigma_2)+'\n')
ghosts.write('Means: '+str(SN_1_mean)+' '+str(SN_2_mean)+' '+str(ratio_1_mean)+
             " +- "+str(ratio_1_mean_err)+' '+str(ratio_2_mean)+" +- "+
             str(ratio_2_mean_err)+' '+str(fmean_1_mean)+' '+str(fmean_2_mean)
             +' '+str(sigma_1_mean)+' '+str(sigma_2_mean))
ghosts.close()


plt.figure(figsize=(8,6))
plt.plot(image_names, SN_1, 'ro', label="SN 1")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(image_names, SN_2, 'ro', label="SN 2")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(image_names, ratio_1, 'bo', label="Ratio 1")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(image_names, ratio_2, 'bo', label="Ratio 2")
plt.legend()
plt.show()

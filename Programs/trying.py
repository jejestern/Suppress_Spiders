"""


2021-10-12
Jennifer Studer <studerje@student.ethz.ch>
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sys import argv, exit
import os
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, find_peaks
from astropy.visualization import simple_norm

# This part takes the argument and saves the folder 
if not len(argv) == 1:
    print("Wrong number of arguments!")
    print("Usage: python find_radius.py")
    print("Exiting...")
    exit()

path = "/home/jeje/Dokumente/Masterthesis/ZirkumstellareScheibe_HD142527"
files = os.listdir(path)

for image_name in files:
    if image_name.endswith(".fits"):      
        img_data = fits.getdata(path + "/" + image_name, ext=0)
        fits.info(path + "/" + image_name)

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

        plt.imshow(int1, cmap='gray', norm = norm)
        plt.show()
"""
        # Position of the ghost in Polar coordinates, where the star is the origin
        positions_gh = [(680.0, 863.0)]
        rad = np.sqrt((positions_gh[0][0]-star_position[0][0])**2 + (positions_gh[0][1]-star_position[0][1])**2)
        phi_gh = np.arctan((positions_gh[0][1]-star_position[0][1])/(positions_gh[0][0]-star_position[0][0]))

        #circ = CircularAperture(star_position, r=round(rad, 0))

        # Transform the ghost position back to the x/y coordinate system
        phi = np.linspace(0, 2*np.pi, 50) + phi_gh
        x_circ = star_position[0][0] + rad* np.cos(phi)
        y_circ = star_position[0][1] + rad* np.sin(phi)
        circ_positions = []
        for i in range(len(x_circ)-1):
            circ_positions.append((x_circ[i],y_circ[i]))


        # We want to find the right radius
        r_apertures = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
        SignaltoNoise = []
        for r_ap in r_apertures:
            print("The chosen aperture radius is ", r_ap)

            # Plot the image
            plt.imshow(int1, cmap='gray', norm = norm)

            circles = CircularAperture(circ_positions, r=r_ap)
            circles.plot(color ='r', lw=1.0)
            annulus_circles = CircularAnnulus(circ_positions, r_in=10, r_out=15)
            annulus_circles.plot(color ='#0547f9', lw=1.0)

            #circ.plot(color ='y', lw=1.0)
            plt.xlim(0, int1.shape[1]-1)
            plt.ylim(0, int1.shape[0]-1)
            #plt.legend()
            plt.colorbar()

            phot_table_circ = aperture_photometry(int1, circles)
            aperture_circ = np.array(phot_table_circ['aperture_sum'])

            signalperpix = []
            f_ap = []

            annulus_circ_masks = annulus_circles.to_mask(method='center')
            for i in range(len(x_circ)-1):
                annulus_circ_data = annulus_circ_masks[i].multiply(int1)

                mask_circ = annulus_circ_masks[i].data
                annulus_circ_data_1d = annulus_circ_data[mask_circ > 0]
                annulus_circ_data_1d.shape

                # Flux from the apertures (like from the report (gisin))
                spixel = sum(annulus_circ_data_1d)/annulus_circ_data_1d.shape[0]
                signalperpix.append(spixel)

                f_ap.append(aperture_circ[i] - circles.area*spixel)

            f_ap = np.array(f_ap)
            #print(signalperpix)
            #print(f_ap)

            fmean = sum(f_ap[1:])/(len(f_ap))
            print("The mean aperture flux is:",fmean)

            sigma = np.sqrt(1/(len(f_ap)-1) * sum((f_ap[1:] - fmean)**2))
            print("Sigma = ", sigma)
            
            SN = (f_ap[0]-fmean)/sigma
            print("We have a signal to noise of ",SN)
            SignaltoNoise.append(SN)

            #plt.show()

        # We plot the radius of the aperture versus the SN
        r_apertures = np.array(r_apertures)
        SignaltoNoise = np.array(SignaltoNoise)

        plt.figure()
        plt.plot(r_apertures, SignaltoNoise, 'o')
        plt.savefig("Aperture_photometry_radius.pdf")
        plt.show()

        # The ideal radius is where the SN has its maximum
        r_where = np.where(max(SignaltoNoise)==SignaltoNoise)[0]
        r_best = r_apertures[r_where]
        print("The ideal radius for the aperture is ", r_best)

"""

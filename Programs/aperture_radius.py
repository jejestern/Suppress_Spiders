"""
This program finds the radius needed for the aperture photometry. It is made 
for the images used in P2, where the ghosts stay at the same position.


Created on 2021-10-12
Jennifer Studer <studerje@student.ethz.ch>
"""
import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import fits
from sys import argv, exit
#import os
from photutils import CircularAperture, CircularAnnulus, aperture_photometry #, find_peaks
#from astropy.visualization import simple_norm

# This part takes the argument and saves the folder 
if not len(argv) == 1:
    print("Wrong number of arguments!")
    print("Usage: python aperture_radius.py")
    print("Exiting...")
    exit()


def aperture_phot(image, norm, position, star_position, r_ap, plot):
    """
    Aperture photometry for a given object which will be compared to its star.
    E.g. for exoplanets, ghosts

    Parameters
    ----------
    image : float32, np.array 
        An intensity image
    norm : astropy.visualization.mpl_normalize.ImageNormalize
        The norm used for plotting the image
    position : list
        Position of the object in the image 
    star_position : list
        Position of the star in the image
    r_ap : int
        Aperture radius used for the aperture photometry
    plot : bool
        True: plot an image of the aperture photometry
        False: do not plot

    Returns
    -------
    fmean : float
        Mean background flux
    sigma : float
        Standard deviation of mean flux
    SN : float
        Signal to noise of the object
    ratio : float
        Contrast between the object and the star

    """
    
    # We transform the position of the target into polar coordinates
    rad = np.sqrt((position[0]-star_position[0][0])**2 + (position[1]-star_position[0][1])**2)
    phi_start = np.arctan((position[1]-star_position[0][1])/(position[0]-star_position[0][0]))
    
    #circ = CircularAperture(star_position, r=round(rad, 0))

    # Transform the target position back to the x/y coordinate system
    phi = np.linspace(0, 2*np.pi, 50) + phi_start
    x_circ = star_position[0][0] + rad* np.cos(phi)
    y_circ = star_position[0][1] + rad* np.sin(phi)
    circ_positions = []
    for i in range(len(x_circ)-1):
        circ_positions.append((x_circ[i],y_circ[i]))
        
    # print("The chosen aperture radius is ", r_ap)

    circles = CircularAperture(circ_positions, r=r_ap)
    annulus_circles = CircularAnnulus(circ_positions, r_in=10, r_out=15)

    # If plot=True, we plot the image with the circles
    if plot==True:
        plt.imshow(image, cmap='gray', norm = norm)
        circles.plot(color ='r', lw=1.0)
        annulus_circles.plot(color ='#0547f9', lw=1.0)
        #circ.plot(color ='y', lw=1.0)
        plt.xlim(0, image.shape[1]-1)
        plt.ylim(0, image.shape[0]-1)
        #plt.legend()
        plt.colorbar()
        plt.show()

    phot_table_circ = aperture_photometry(image, circles)
    aperture_circ = np.array(phot_table_circ['aperture_sum'])

    signalperpix = []
    f_ap = []

    annulus_circ_masks = annulus_circles.to_mask(method='center')
    for i in range(len(x_circ)-1):
        annulus_circ_data = annulus_circ_masks[i].multiply(image)

        mask_circ = annulus_circ_masks[i].data
        annulus_circ_data_1d = annulus_circ_data[mask_circ > 0]
        annulus_circ_data_1d.shape

        # Flux from the apertures (like from the report (gisin))
        spixel = sum(annulus_circ_data_1d)/annulus_circ_data_1d.shape[0]
        signalperpix.append(spixel)

        f_ap.append(aperture_circ[i] - circles.area*spixel)

    f_ap = np.array(f_ap)

    fmean = sum(f_ap[1:])/(len(f_ap))
    sigma = np.sqrt(1/(len(f_ap)-1) * sum((f_ap[1:] - fmean)**2))
    SN = (f_ap[0]-fmean)/sigma
    
    # To calculate the ratio between star and object
    apertures = CircularAperture(star_position, r=r_ap)
    annulus_aperture = CircularAnnulus(star_position, r_in=10, r_out=15)
    phot_table = aperture_photometry(image, apertures)
    aperture = np.array(phot_table['aperture_sum'])
    annulus_masks = annulus_aperture.to_mask(method='center') 
    annulus_data = annulus_masks[0].multiply(image)
    mask = annulus_masks[0].data
    annulus_data_1d = annulus_data[mask > 0]
    spixel_star = sum(annulus_data_1d)/annulus_data_1d.shape[0]
    f_ap_star = aperture - apertures.area*spixel_star
    ratio = f_ap[0]/f_ap_star
    
    return fmean, sigma, SN, ratio

def find_radius(image, norm, position, star_position):
    """
    Chooses a radius for the aperture photometry by increasing the Signal to 
    Noise.

    Parameters
    ----------
    image : float32, np.array 
        An intensity image
    norm : astropy.visualization.mpl_normalize.ImageNormalize
        The norm used for plotting the image
    position : list
        Position of the object in the image 
    star_position : list
        Position of the star in the image

    Returns
    -------
    best_radius : int
        Aperture radius for which the Signal to Noise is maximal

    """
    
    radi = np.arange(1, 10) 
    SignaltoNoise = []
    for r_ap in radi:
        fmean, sigma, SN, ratio = aperture_phot(image, norm, position, star_position, r_ap, plot=False)
        SignaltoNoise.append(SN)

    # We plot the radius of the aperture versus the SN
    radi = np.array(radi)
    SignaltoNoise = np.array(SignaltoNoise)

    plt.figure()
    plt.plot(radi, SignaltoNoise, 'o')
    #plt.savefig("Aperture_photometry_radius.pdf")
    plt.show()

    # The ideal radius is where the SN has its maximum
    r_where = np.where(max(SignaltoNoise)==SignaltoNoise)[0]
    best_radius = radi[r_where][0]
    # print("The ideal radius for the aperture is ", best_radius)
    return best_radius

"""
path = "/home/jeje/Dokumente/Masterthesis/ZirkumstellareScheibe_HD142527"
files = os.listdir(path)

for image_name in files:
    if image_name.endswith("cyc129_normft_1.fits"):
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

        # Position of the ghost in Polar coordinates, where the star is the origin
        positions_gh = [(680.0, 863.0)]
        
        fmean, sigma, SN= aperture_phot(int1, norm, positions_gh[0], star_position, 4.0, plot=True)
        print("The mean aperture flux is:",fmean)
        print("Sigma = ", sigma)
        print("We have a signal to noise of ",SN)
        
        radius_ap = find_radius(int1, norm, positions_gh[0], star_position)
        print("The ideal radius for the aperture is ", radius_ap)
"""
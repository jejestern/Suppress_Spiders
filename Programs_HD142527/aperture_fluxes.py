#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:35:11 2021

@author: jeje
"""
import numpy as np
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture, EllipticalAnnulus, aperture_photometry 
from transformations_functions import xy_to_rphi


def aperture_flux_image(image, object_positions):
    
    # Calculate the aperture sum of the model planet
    aperture_rad = 6.0
    aper = CircularAperture(object_positions, r=aperture_rad) 
    annu = CircularAnnulus(object_positions, r_in=10, r_out=15)
    
    phot_table = aperture_photometry(image, aper)
    aperture = np.array(phot_table['aperture_sum'])
    annu_masks = annu.to_mask(method='center') 
    annu_data = annu_masks.multiply(image)
    mask = annu_masks.data
    annu_data_1d = annu_data[mask > 0]
    spixel = sum(annu_data_1d)/annu_data_1d.shape[0]
    f_ap = aperture - aper.area*spixel
    
    return f_ap[0], aper, annu

def aperture_flux_warped(im_warped, w_shape, r_start, aspect_value, object_positions):
    
    aperture_rad = 6.0
    
    # Calculate object position in polar coordinates
    r_pos, phi_pos = xy_to_rphi(object_positions[0]-511, object_positions[1]-511)
    
    # Calculate the aperture and annulus for plotting
    aper_draw = EllipticalAperture([phi_pos, r_pos], aperture_rad*aspect_value, 
                                   aperture_rad)
    annu_draw = EllipticalAnnulus([phi_pos, r_pos], 10*aspect_value, 
                                  15*aspect_value, 15, 10)
    
    # Calculate the aperture sum of the model planet in the warped image
    warped_pos = [phi_pos/(2*np.pi)*w_shape[0], r_pos-r_start]
    aper = CircularAperture(warped_pos, aperture_rad)
    annu = CircularAnnulus(warped_pos, 10, 15)

    phot_table = aperture_photometry(im_warped, aper)
    aperture = np.array(phot_table['aperture_sum'])
    annu_masks = annu.to_mask(method='center') 
    annu_data = annu_masks.multiply(im_warped)
    mask = annu_masks.data
    annu_data_1d = annu_data[mask > 0]
    spixel = sum(annu_data_1d)/annu_data_1d.shape[0]
    f_ap = aperture - aper.area*spixel
    
    return f_ap[0], aper_draw, annu_draw




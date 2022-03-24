#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:10:46 2022

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('aperture_fluxes_ghost1.npz')
aper_origin = data['name1']
aper_warped = data['name2']  
aper_flat = data['name3']
aper_cfreq = data['name4'] 
aper_lfreq = data['name5']     
        
x_len = np.arange(len(aper_origin))

# We plot the different apertures
plt.figure()
plt.plot(x_len, aper_origin, 'x', label="Original Image")
plt.plot(x_len, aper_warped, 'x', label="Warped")
plt.plot(x_len, aper_flat, 'x', label="Flattened")
plt.plot(x_len, aper_cfreq, 'x', label="Suppressing central radial frequency")
plt.plot(x_len, aper_lfreq, 'x', label="Suppressing lower frequencies")
plt.xlabel("Images from HD142527")
plt.ylabel("Aperture flux of ghost 1")
plt.legend()
plt.tight_layout()
plt.savefig("Ghost1_apertures.pdf")
plt.show()

# We plot the percentual aperture change, starting by the flattened image
plt.figure()
plt.plot(x_len, 100/aper_flat*aper_flat, 'x', label="Flattened")
plt.plot(x_len, 100/aper_flat*aper_cfreq, 'x', label="Suppressing central radial frequency")
plt.plot(x_len, 100/aper_flat*aper_lfreq, 'x', label="Suppressing lower frequencies")
plt.xlabel("Images from HD142527")
plt.ylabel("Aperture change due to suppression: ghost 1  [%]")
plt.legend()
plt.tight_layout()
plt.savefig("Ghost1_apertures_perc.pdf")
plt.legend()
plt.show()
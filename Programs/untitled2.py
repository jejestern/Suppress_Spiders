#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:44:30 2021

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/cyc_161_R250to450/radtophi_img.txt").reshape(200, 200)

plt.imshow(data, origin='lower', cmap='gray')
plt.colorbar()
plt.show()



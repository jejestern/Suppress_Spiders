#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We use the transformation data to examine the star and its surrounding
Atention: the x-axis is the vertical axis and the y-axis is the horizontal axis
in this case.

Created on 2021-10-26
Jennifer Studer <studerje@student.ethz.ch>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def e_func(x, a, b, c):

    return a * np.exp(-b * x) + c



data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/cyc_161_R250to450/radtophi_img.txt").reshape(200, 200)


plt.imshow(data, origin='lower', cmap='gray')
plt.colorbar()
plt.show()

x = np.arange(len(data[0]))
plt.figure()
plt.plot(x, data[0], 'b.', label="Intensity distribution for R smallest")
plt.legend()
plt.show()

# We plot the intensity along the y axis and do a fit
spike1 = np.where(data[0] == max(data[0]))[0][0]

y = np.arange(len(data[:,spike1]))

## Fitting an exponential
popt, pcov = curve_fit(e_func, y, data[:,spike1])
print(popt)

plt.figure()
plt.plot(y, e_func(y, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.plot(y, data[:,spike1], 'b-', label="intensity distribution of longest speckle")
plt.legend()
plt.show()

data[:,spike1] = data[:,spike1] - e_func(y, *popt)
plt.imshow(data, origin='lower', cmap='gray')
plt.colorbar()
plt.show()
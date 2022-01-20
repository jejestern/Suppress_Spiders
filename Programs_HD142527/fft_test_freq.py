#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:29:46 2022

@author: jeje
"""

import numpy as np
import matplotlib.pyplot as plt



t = np.linspace(0, 2*np.pi, 1000, endpoint=True)
f = 3.0 # Frequency in Hz
A = 100.0 # Amplitude in Unit
s = A * np.sin(2*np.pi*f*t) # Signal

plt.figure()
plt.plot(t,s)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude ($Unit$)')
plt.show()


# Fourier transform the function
fft = np.fft.fftshift(np.fft.fft(s))

N = int(len(fft)/2+1)



dt = t[1] - t[0]
fa = 1.0/dt # scan frequency
print('dt=%.5fs (Sample Time)' % dt)
print('fa=%.2fHz (Frequency)' % fa)

X = np.linspace(0, fa/2, 1000, endpoint=True)

plt.figure()
plt.plot(X, np.abs(fft))
plt.xlabel('Frequency ($Hz$)')
plt.show()


freq = np.fft.fftfreq(1000, dt)
freq = np.fft.fftshift(freq)
plt.figure()
plt.plot(freq, np.abs(fft))
plt.xlabel('Frequency ($Hz$)')
plt.show()

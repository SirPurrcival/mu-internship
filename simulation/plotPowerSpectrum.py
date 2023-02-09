#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:42:19 2023

@author: meowlin
"""

##

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def plotPowerSpectrum(data, resolution):
    
    (S, f) = plt.psd(data, Fs=1/resolution)
    
    plt.semilogy(f, S)
    plt.xlim([0, 100])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    
    # ps = np.abs(np.fft.fft(data))**2

    # time_step = resolution
    # freqs = np.fft.fftfreq(data.size, time_step)
    # idx = np.argsort(freqs)

    # plt.plot(freqs[idx], ps[idx])
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io as sp
import numpy as np
import pylab as plt

monkey = 2
session = 3
# contrast = 
trial = 3

def plottrial(trialnr):
    data = sp.loadmat('M'+str(monkey)+'_session_'+str(session)+'.mat')

    data = data['alldata']

    data = data[0, 0]

    actual_data = data[0]
    rfdata = data[1]
    probe1_contacts = data[2]
    probe2_contacts = data[3]
    probe3_contacts = data[4]
    eyedata = data[5]

    cont_data = actual_data[0, 0]
    CSD = cont_data[0, 0][0]
    probe1_contrast = cont_data[0, 0][1]
    probe2_contrast = cont_data[0, 0][2]
    probe3_contrast = cont_data[0, 0][3]

    trial = CSD[0,0][3][0, trialnr-1]
    time = CSD[0,0][2][0, trialnr-1]
    print(np.shape(time))
    print(np.shape(trial))

    plt.figure()
    plt.subplot(311)
    plt.imshow(trial[0:15], aspect='auto')
    plt.subplot(312)
    plt.imshow(trial[16:31], aspect='auto')
    plt.subplot(313)
    plt.imshow(trial[32:47], aspect='auto')
    plt.xticks(np.linspace(0, np.shape(time)[1], 8), 
                np.round(np.linspace(0, np.shape(time)[1], 8),2))
    plt.show()
    
plottrial(trial)

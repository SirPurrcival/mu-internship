#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:30:17 2022

@author: meowlin

Data preprocessing script for the internship
"""

import scipy.io as sp
import numpy as np


tmp = sp.loadmat("M1_session_1.mat")["alldata"]
data = tmp[0,0]
np.shape(tmp)
np.shape(data[0])
data_tmp = data[0]
np.shape(data_tmp)
dt = data_tmp[0,1]
np.shape(dt)
dt = dt[0,0]
print(type(dt))
dt[0]
dt[1]

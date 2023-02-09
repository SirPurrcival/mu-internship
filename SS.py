#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:44:14 2023

@author: fiona
"""

import numpy as np

SS = np.array([[1.73, 0.53, 0.48, 0.57, 0.78, 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   ],
               [0   , 0.36, 1.49, 0.86, 1.31, 0.34, 1.39, 0.69, 0.91, 0.74, 1.32, 0.53, 0   , 0   , 0   , 0   , 0   ],
               [0.37, 0.48, 0.68, 0.42, 0.41, 0.56, 0.68, 0.42, 0.41, 0.2 , 0.79, 0   , 0   , 0   , 0   , 0   , 0   ],
               [0.47, 0.31, 0.5 , 0.15, 0.52, 0.3 , 0.5 , 0.15, 0.52, 0.22, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
               [0   , 0.28, 0.18, 0.32, 0.37, 0.29, 0.18, 0.32, 0.37, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
               [0   , 0.78, 1.39, 0.69, 0.91, 0.83, 1.29, 0.51, 0.51, 0.63, 1.25, 0.52, 0.91, 0.96, 0   , 0   , 0   ],
               [0   , 0.56, 0.68, 0.42, 0.41, 0.64, 0.68, 0.42, 0.41, 0.73, 0.94, 0.42, 0.41, 0   , 0   , 0   , 0   ],
               [0.39, 0.3 , 0.5 , 0.15, 0.52, 0.29, 0.5 , 0.15, 0.52, 0.28, 0.45, 0.28, 0.52, 0   , 0   , 0   , 0   ],
               [0   , 0.29, 0.18, 0.32, 0.37, 0.29, 0.18, 0.32, 0.37, 0   , 0.18, 0.33, 0.37, 0   , 0   , 0   , 0   ],
               [0.76, 0.47, 1.25, 0.52, 0.91, 0.38, 1.25, 0.52, 0.91, 0.75, 1.2 , 0.52, 1.31, 0.4 , 2.5 , 0.52, 1.31],
               [0   , 0   , 0.51, 0   , 0   , 0   , 0.94, 0.42, 0.41, 0.81, 1.19, 0.41, 0.41, 0.81, 1.19, 0.41, 0.41],
               [0.31, 0.25, 0   , 0.39, 0   , 0.28, 0.45, 0.28, 0.52, 0.27, 0.4 , 0.4 , 0.52, 0.27, 0.4 , 0.4 , 0.52],
               [0   , 0   , 0   , 0   , 0   , 0.29, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37],
               [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.23, 2.5 , 0.52, 1.31, 0.94, 3.8 , 0.52, 1.31],
               [0   , 0.81, 0   , 0   , 0   , 0.81, 0   , 0   , 0   , 0.81, 1.19, 0.41, 0.41, 0.81, 1.19, 0.41, 0.41],
               [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.27, 0.4 , 0.4 , 0.52, 0.27, 0.4 , 0.4 , 0.52],
               [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.28, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37]])
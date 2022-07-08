# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:56:58 2022

@author: dugga
"""
import matplotlib.pyplot as plt
#Brownian Motion
# N(0,sqrt(step_len))
paths=1000

steps= 2000
T=100
import scipy.stats as ss
import numpy as np
X0 = np.zeros((paths,1)) #all paths start from 0
T_arr, step_len = np.linspace(0, T, steps ,retstep=True)
rand_nos = ss.norm.rvs(loc=0, scale=np.sqrt(step_len), size = (paths,steps-1))

X = np.concatenate((X0,rand_nos), axis=1).cumsum(1)

plt.plot(T_arr,X.T)

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:07:33 2022

@author: Lenovo
"""

###################################################
##############IMPORT LIBRARIES#####################
################################################### 
import numpy as np
import matplotlib.pyplot as plt
import math

def lorenz(xEU,yEU,zEU,par):
    x_dot = par[0]*(yEU[i]-xEU[i])
    y_dot = par[2]*xEU[i] - xEU[i]*zEU[i] - yEU[i]
    z_dot = xEU[i]*yEU[i] - par[1]*zEU[i]
    return x_dot,y_dot,z_dot


num_steps=12000 #number of time steps (initial time step not included)
parset=0
eps = 1E-3
L0=[9.,10.,20.] #initial condition 1
L01=[9. + eps,10.,20.] #initial condition 2
dt=0.005 #time step 

    #parametri sigma, b ed r
set_par1=[10.,8./3.,28.] 
set_par2=[10.,8./3.,9.]
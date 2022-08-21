# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:58:03 2022

@author: Lenovo
"""
from lorenz import lorenz
from lorenz import perturbation
from lorenz import difference
from lorenz import RSME
from plots import xzgraph
from plots import plot_difference
from plots import plot_rsme
from scipy.integrate import odeint
import numpy as np
import configparser
#from configparser import ConfigParser


config = configparser.ConfigParser()
config.read('config.ini')


#-----------------------Parameters-------------------------#
#Canonical choice of the system parameters, given by Lorenz

sigma = 10.

b = 8./3.

r1 = 28. #chaotic solution

r2 = 9.  #non-chaotic solution


#-----------------Integration parameters-------------------#

num_steps = config.get('Integration settings', 'num_steps')
num_steps = int(num_steps)

dt = config.get('Integration settings', 'dt')
dt = float(dt)

IC0 =  [9.,10.,20.] #initial condition

eps = [1E-5,1E-3,1.0] #perturbations

t = np.linspace(0,num_steps,num_steps+1)*dt #time variable

#-----------------------Integration------------------------#

IC = perturbation(IC0,eps) #perturbed initial condition


sol_1 = odeint(lorenz,IC[0,:],t,args=(sigma,b,r1)) #chaotic solution
sol_2 = odeint(lorenz,IC[0,:],t,args=(sigma,b,r2)) #non-chaotic solution

sol_1_per1 = odeint(lorenz,IC[1,:],t,args=(sigma,b,r1)) #chaotic solution
sol_2_per2 = odeint(lorenz,IC[1,:],t,args=(sigma,b,r2)) #non-chaotic solution

#-------------------------Analysis-------------------------#

delta_x = difference(sol_1, sol_1_per1)

error = RSME(sol_1, sol_1_per1)




#------------------------Plotting--------------------------#

xzgraph(sol_1,r1)
xzgraph(sol_2,r2)

plot_difference(delta_x,t)

plot_rsme(error,t)

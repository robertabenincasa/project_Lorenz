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
from plots import plot_3dsolution
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

f1 = 0.

f2 = 1.

#-----------------Integration parameters-------------------#

num_steps = config.get('Integration settings', 'num_steps')
num_steps = int(num_steps)

dt = config.get('Integration settings', 'dt')
dt = float(dt)

IC0 =  [9.,10.,20.] #initial condition

eps = [0.,1E-5,1E-3,1.] #perturbations

t = np.linspace(0,num_steps,num_steps+1)*dt #time variable


#-----------------------Integration------------------------#

IC = perturbation(IC0,eps) #perturbed initial condition

sol_1 = np.zeros((num_steps + 1, 3, len(IC)))
sol_2 = np.zeros((num_steps + 1, 3, len(IC)))

delta_x = np.zeros((num_steps+1, 2))
error = np.zeros((num_steps+1, len(IC)))
pred_time = np.zeros(len(IC))

#Unperturbed solutions

for i in range(len(IC)):
    
    sol_1[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r1,f1)) #chaotic solution
    sol_2[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r2,f1)) #non-chaotic solution

#First perturbation

#sol_1_per1 = odeint(lorenz,IC[1,:],t,args=(sigma,b,r1,f1)) #chaotic solution
#sol_2_per1 = odeint(lorenz,IC[1,:],t,args=(sigma,b,r2,f1)) #non-chaotic solution





#-------------------------Analysis-------------------------#

delta_x[:,0] = difference(sol_1[:,:,0], sol_1[:,:,1])
delta_x[:,1] = difference(sol_2[:,:,0], sol_2[:,:,1])

for i in range(1,len(IC)):
    
    error[:,i] = RSME(sol_1[:,:,0], sol_1[:,:,i])
    for m in range(num_steps): 
        if error[m,i] > 0.5:
            pred_time[i] = m * dt
            break 



#------------------------Plotting--------------------------#

    
xzgraph(sol_1[:,:,0],r1)
xzgraph(sol_2[:,:,0],r2)

plot_3dsolution(sol_1[:,:,0],r1)
plot_3dsolution(sol_2[:,:,0],r2)

plot_difference(delta_x[:,0],t, r1)
plot_difference(delta_x[:,1],t, r2)

for i in range(1,len(IC)):
   
    plot_rsme(error[:,i],t, r1, eps[i], pred_time[i])

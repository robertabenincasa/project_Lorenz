# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:58:03 2022

@author: roberta benincasa
"""
from lorenz import (lorenz, perturbation, difference,
                    RMSE, prediction, read_parameters)

from plots import (xzgraph, plot_difference, plot_rmse,
                   plot_3dsolution, plot_animation,plot_ensemble)

from tabulate import tabulate
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from pandas import DataFrame
import dataframe_image as dfi
import configparser



config = configparser.ConfigParser()
config.read('config.ini')


#-----------------------Parameters-------------------------#


sigma1 = config.get('Parameters', 'sigma')
sigma = float(sigma1)

b1 = config.get('Parameters', 'b')
b = float(b1)

r_1 = config.get('Parameters', 'r1') #chaotic solution
r1 = float(r_1)

r_2 =  config.get('Parameters', 'r2') #non-chaotic solution
r2 = float(r_2)

#-----------------Integration parameters-------------------#

num_steps0 = config.get('Integration settings', 'num_steps')
num_steps = int(num_steps0)

num_steps2 = config.get('Integration settings', 'num_steps1')
num_steps1 = int(num_steps2)

dt1 = config.get('Integration settings', 'dt')
dt = float(dt1)

N1 = config.get('Integration settings', 'N')
N = int(N1)

IC01 = config.get('Initial condition', 'IC') #initial condition
IC0 = read_parameters(IC01)

eps1 = config.get('Perturbations', 'eps') #perturbations
eps = read_parameters(eps1)
    

t = np.linspace(0,num_steps,num_steps)*dt #time variable
t1 = np.linspace(0,num_steps1,num_steps1)*dt

#-----------------------Integration------------------------#

#Initializing arrays

#The following are the solution for each time step,
#for each variable and for each IC
sol_1 = np.zeros((num_steps , 3, len(eps)+1)) 
#chaotic solution 
sol_2 = np.zeros((num_steps , 3, len(eps)+1))
#non-chaotic solution 


IC = perturbation(IC0,eps) #perturbed initial conditions

#Integrating

#Solutions for each value of r and for each IC

for i in range(len(eps)+1):
    
    sol_1[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r1)) 
    #chaotic solution
    sol_2[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r2))
    #non-chaotic solution


#-------------------------Analysis-------------------------#

#Initializing arrays

delta_x = np.zeros((num_steps, 2)) 
#The difference is performed only between the solution of the unperturbed 
#case and the one of the first perturbed case, as a preliminary analysis.
#The difference is calculated for both chaotic and non-chaotic solution.

error = np.zeros((num_steps, len(eps)))
pred_time = np.zeros(len(eps))
#The RMSE and the prediction time are calculated for each perturbed case
#with r = 28

                 
 
#Calculating

delta_x[:,0] = difference(sol_1[:,:,0], sol_1[:,:,1]) 
delta_x[:,1] = difference(sol_2[:,:,0], sol_2[:,:,1])

for i in range(1,len(eps)+1): 
    
    error[:,i-1] = RMSE(sol_1[:,:,0], sol_1[:,:,i])
    
pred_time = prediction(error, num_steps, dt, eps)

#--------------------------Ensemble------------------------------#
#Same procedure but with an ensemble of perturbation
eps1 = np.zeros(N)

sol_ens = np.zeros((num_steps1 , 3, len(eps1)))
error_ens = np.zeros((num_steps1, len(eps1)))
np.random.seed(44)

for k in range(len(eps1)):
       
        eps1[k] = np.random.random()*1.50 - 0.75

IC_ens = perturbation(IC0,eps1)

for i in range(1,len(eps1)):
    
    sol_ens[:,:,i] = odeint(lorenz,IC_ens[i,:],t1,args=(sigma,b,r1)) 
    error_ens[:,i-1] = RMSE(sol_1[0:num_steps1,:,0], sol_ens[:,:,i])

#R is the mean of the RMSEs and L is the RMSE of the mean
R = np.mean(error_ens,1)    
pred_times = np.zeros(2)
sol_ave = np.mean(sol_ens,2)
L = RMSE(sol_1[0:num_steps1,:,0], sol_ave[:,:])
errors = [L, R]
for j in errors:
    
    for m in range(num_steps1): 

        if j[m] > 0.5:
    
            pred_times[0] = m * dt 
    
            break 
    

#------------------------Plots & Tables--------------------------#

#plotting both chaotic and non-chaotic solution for
#the unpertubed case in the x,z plane

xzgraph(sol_1[:,:,0],r1) 
xzgraph(sol_2[:,:,0],r2) 

#3D-plotting non-chaotic solution for the unpertubed case 

plot_3dsolution(sol_2[:,:,0],r2)

#3D animation of the chaotic solution for the unperturbed and a 
#perturbed case 
print('------This operation may require a few seconds-----')
plot_animation(sol_1[:,:,0],sol_1[:,:,3],r1,eps[2])


plot_difference(delta_x[:,0],t, r1) 
plot_difference(delta_x[:,1],t, r2)

for i in range(len(eps)): 
   
    plot_rmse(error[:,i],t, r1, eps[i], pred_time[i])
    
plot_ensemble(L,R,t1)


#creating a table with the values of the perturbation and
# of the corresponding prediction times 

data = np.column_stack((eps, pred_time))

col_names = ["Perturbation", "Prediction time"]

print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

data1 = np.column_stack((pred_times[0], pred_times[1]))
col_names1 = ["Prediction time L", "Prediction time R"]
print(tabulate(data1, headers=col_names1, tablefmt="fancy_grid"))


df = pd.DataFrame(data, columns=['Perturbation','Prediction time'])
dfi.export(df, 'table.png')



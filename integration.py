# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:58:03 2022

@author: roberta benincasa
"""
import configparser
import numpy as np
import pandas as pd
import dataframe_image as dfi
from tabulate import tabulate
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from lorenz import (lorenz, perturbation, difference, func,
                    RMSE, prediction, read_parameters, ensemble)

from plots import (xzgraph, plot_difference, plot_rmse,
                   plot_3dsolution, plot_animation,plot_ensemble, 
                   plot_ensemble_trajectories, pred_time_vs_perturbation)


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

dt1 = config.get('Integration settings', 'dt')
dt = float(dt1)

N1 = config.get('Integration settings', 'N')
N = int(N1)

IC01 = config.get('Initial condition', 'IC') #initial condition
IC0 = read_parameters(IC01)

eps1 = config.get('Perturbations', 'eps') #perturbations
eps = read_parameters(eps1)
    

t = np.linspace(0,num_steps,num_steps)*dt #time variable


#-----------------------Integration------------------------#


#The following are the solution for each time step,
#for each variable and for each IC

sol_1 = np.zeros((num_steps , 3, len(eps)+1)) #chaotic solution

sol_2 = np.zeros((num_steps , 3, len(eps)+1)) #non-chaotic solution
 


IC = perturbation(IC0,eps) #perturbed initial conditions


#Solutions for each value of r and for each IC

for i in range(len(eps)+1):
    
    sol_1[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r1)) 
    #chaotic solution
    sol_2[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r2))
    #non-chaotic solution


#-------------------------ANALYSIS-------------------------#

#The difference is performed only between the solution of the unperturbed 
#case and the one of the first perturbed case, as a preliminary analysis.
#The difference is calculated for both chaotic and non-chaotic solution.

delta_x = np.zeros((num_steps, 2)) 

#The RMSE and the prediction time are calculated for each perturbed case
#with r = 28 because it would be trivial for r = 9.

error = np.zeros((num_steps, len(eps)))
pred_time = np.zeros(len(eps))
                 

delta_x[:,0] = difference(sol_1[:,:,0], sol_1[:,:,3]) 
delta_x[:,1] = difference(sol_2[:,:,0], sol_2[:,:,3])

for i in range(1,len(eps)+1): 
    
    error[:,i-1] = RMSE(sol_1[:,:,0], sol_1[:,:,i])
    
pred_time = prediction(error, num_steps, dt, eps)

#--------------------------ENSEMBLE------------------------------#

#Same procedure but with an ensemble of perturbations: showing how to improve
#the prediction!

eps_ens = np.zeros(N)
pred_time_ens = np.zeros(N)
sol_ens = np.zeros((num_steps , 3, N))
error_ens = np.zeros((num_steps, N))

np.random.seed(42)

for k in range(N):
       
    eps_ens[k] = np.random.random()*1.50 - 0.75

#eps_ens = np.logspace(-5, 0., num=N, base=10.0)

IC_ens = perturbation(IC0,eps_ens)

for i in range(N):
    
    sol_ens[:,:,i] = odeint(lorenz,IC_ens[i,:],t,args=(sigma,b,r1)) 
    error_ens[:,i] = RMSE(sol_1[:,:,0], sol_ens[:,:,i])


#R is the mean of the RMSEs and L is the RMSE of the mean.
#The aim is to compare the 2 and show how introducing an ensemble of simulations
#allows to halve the RMSE with respect to the one relative to a single simulation.

R = np.mean(error_ens, 1) 
   
spread, sol_ave = ensemble(sol_ens)        

L = RMSE(sol_1[:,:,0], sol_ave[:,:])

pred_times = np.zeros(2)

errors = [L, R]

for j in errors:
    
    for m in range(num_steps): 

        if j[m] > 0.5:
    
            pred_times[0] = m * dt 
    
            break 
    

#------------------------Plots & Tables--------------------------#

path = config.get('Paths to files', 'path')

#plotting both chaotic and non-chaotic solution for
#the unpertubed case in the x,z plane

xzgraph(sol_1[:,:,0],r1) 
xzgraph(sol_2[:,:,0],r2) 

plot_3dsolution(sol_2[:,:,0],r2)

#3D animation of the chaotic solution for the unperturbed and a 
#perturbed case 
print('\n')
print('---------------Preparing the animation--------------')
print('------This operation may require a few seconds------')

plot_animation(sol_1[:,:,0],sol_1[:,:,4],r1,eps[4])



#Plotting the results of the analysis:

plot_difference(delta_x[:,0],delta_x[:,1],t,eps[3]) 

for i in range(len(eps)): 
   
    plot_rmse(error[:,i],t, r1, eps[i], pred_time[i])
 
#Fitting: predictability time vs applied perturbation

popt, pcov = curve_fit(func, np.log10(eps), pred_time)

fit = func(np.log10(eps),*popt)

pred_time_vs_perturbation(pred_time, eps, fit, popt)

#Plotting the results of hte ensemble analysis 
plot_ensemble(L,R,t)
plot_ensemble_trajectories(sol_ave,spread,t)


#creating a table with the values of the perturbation and
# of the corresponding prediction times 

#---------------------Printing to terminal:------------------------


print('\n')
print('Lorenz system with r = 28:')
print('\n')

data = np.column_stack((eps, pred_time))
col_names = ["Perturbation", "Predictability time"]
print('Single forecast:')
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

data1 = np.column_stack((pred_times[0], pred_times[1]))
col_names1 = ["Prediction time L", "Predictability time R"]
print('Ensemble forecast:')
print(tabulate(data1, headers=col_names1, tablefmt="fancy_grid"))

#-------------------Saving to csv and to png files:-----------------

df = pd.DataFrame(data = {'Perturbation': eps,'Predictability time': pred_time})
#df.to_csv(path + '/Table_pred_time.csv', index = False, sep = " ", decimal= ",",
#          columns=['Perturbation','Predictability time'])
dfi.export(df, path + '/table_predtime.png',fontsize = 30)

col1 = pred_times[0]
col2 = pred_times[1]

df1 = pd.DataFrame(data = {'Predictability time L': col1, 
                           'Predictability time R': col2},index= [1])
#df1.to_csv(path + '/Table_L&R.csv', index = False, sep = " ", decimal= ",",
#           columns=['Predictability time L', 'Predictability time R'])
dfi.export(df1, path + '/table_LR.png',fontsize = 30)

print('\n')
print('Plots and tables are now available in the folder: output')

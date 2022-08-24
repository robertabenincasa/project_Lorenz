# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:58:03 2022

@author: roberta benincasa
"""
from lorenz import (lorenz, perturbation, difference, RMSE, prediction)
from plots import (xzgraph, plot_difference, plot_rmse, plot_3dsolution)
from tabulate import tabulate
import scipy.integrate
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from pandas import DataFrame
import configparser
#from configparser import ConfigParser


config = configparser.ConfigParser()
config.read('config.ini')


#-----------------------Parameters-------------------------#
#Canonical choice of the parameters of the Lorenz system,

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

t = np.linspace(0,num_steps,num_steps)*dt #time variable


#-----------------------Integration------------------------#

IC = perturbation(IC0,eps) #perturbed initial conditions

#Initializing arrays

sol_1 = np.zeros((num_steps , 3, len(eps))) #chaotic solution for each IC
sol_2 = np.zeros((num_steps , 3, len(eps))) #non-chaotic solution for each IC

sol_1_f = np.zeros((num_steps , 3))
sol_2_f = np.zeros((num_steps , 3))

#Integrating
#Solutions for each value of r and for each IC
for i in range(len(eps)):
    
    sol_1[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r1,f1)) #chaotic solution
    sol_2[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r2,f1)) #non-chaotic solution

    


#sol_1_f[:,:] = odeint(lorenz,IC[0,:],t,args=(sigma,b,r1,f2)) #chaotic solution
#sol_2_f[:,:] = odeint(lorenz,IC[0,:],t,args=(sigma,b,r2,f2)) #non-chaotic solution


#-------------------------Analysis-------------------------#

#Initializing arrays

delta_x = np.zeros((num_steps, 2)) #the difference was performed only 
                                     #between the solution of unperturbed 
                                     #case and the one of the first perturbed 
                                     #case, as a preliminary analysis

error = np.zeros((num_steps, len(eps))) #the rmse was calculated for each
                                         #perturbed case with r = 28
pred_time = np.zeros(3)                  #the same for the prediction time 
 
#Calculating

#the difference was calculated for both chaotic and non-chaotic solution

delta_x[:,0] = difference(sol_1[:,:,0], sol_1[:,:,1]) 
delta_x[:,1] = difference(sol_2[:,:,0], sol_2[:,:,1])

for i in range(1,len(eps)): 
    
    error[:,i] = RMSE(sol_1[:,:,0], sol_1[:,:,i])
    
    

pred_time = prediction(error, num_steps, dt, eps)


#------------------------Plots & Tables--------------------------#

    
xzgraph(sol_1[:,:,0],r1) #plotting both chaotic and non-chaotic solution for
xzgraph(sol_2[:,:,0],r2) #for the unpertubed case in the x,z plane

#xzgraph(sol_2_f[:,:],r2)
#xzgraph(sol_1_f[:,:],r1)


plot_3dsolution(sol_1[:,:,0],r1)
plot_3dsolution(sol_2[:,:,0],r2)

plot_difference(delta_x[:,0],t, r1) #plot the calculated differences
plot_difference(delta_x[:,1],t, r2)

for i in range(1,len(eps)): #plot the rmse for each IC
   
    plot_rmse(error[:,i],t, r1, eps[i], pred_time[i-1])


#creating a table with the values of the perturbation and of the corresponding
#prediction times

data = np.column_stack((eps[1:len(eps)], pred_time))

col_names = ["Perturbation", "Prediction time"]

print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))


df = pd.DataFrame(data, columns=['Perturbation','Prediction time'])
print(df)

#---------------------------Testing--------------------------#

def test_valid_IC():
    
    for i in range(3):  
    
        assert IC0[i] == IC[0,i]

def test_valid_IC_1():
    
    for i in range(1,len(eps)):
        
        for m in range(2,3):
         
            assert IC0[m] == IC[i,m]
        
 
def test_diff1():
    
    delta_x = difference(sol_1[:,:,0],sol_1[:,:,0])
    
    for i in range(num_steps):
        
        assert delta_x[i] == 0.    
 
    
def test_RMSE():
    
    for i in range(num_steps):
        
        for j in range(1,len(eps)):  
            
            assert error[i,j] >= 0.
            
            
def test_RMSE1():
    
    rmse = RMSE(sol_1[:,:,0],sol_1[:,:,0])
    
    for i in range(num_steps):
        
        assert rmse[i] == 0.
    
            

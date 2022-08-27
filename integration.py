# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:58:03 2022

@author: roberta benincasa
"""
from lorenz import (lorenz, perturbation, difference,
                    RMSE, prediction, read_parameters)

from plots import (xzgraph, plot_difference, plot_rmse,
                   plot_3dsolution, plot_animation, animate)

from tabulate import tabulate
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from pandas import DataFrame
import configparser



config = configparser.ConfigParser()
config.read('config.ini')


#-----------------------Parameters-------------------------#


sigma = config.get('Parameters', 'sigma')
sigma = float(sigma)

b = config.get('Parameters', 'b')
b = float(b)

r1 = config.get('Parameters', 'r1') #chaotic solution
r1 = float(r1)

r2 =  config.get('Parameters', 'r2') #non-chaotic solution
r2 = float(r2)

#-----------------Integration parameters-------------------#

num_steps = config.get('Integration settings', 'num_steps')
num_steps = int(num_steps)

dt = config.get('Integration settings', 'dt')
dt = float(dt)

IC0 = config.get('Initial condition', 'IC') #initial condition
IC0 = read_parameters(IC0)

eps = config.get('Perturbations', 'eps') #perturbations
eps = read_parameters(eps)
    

t = np.linspace(0,num_steps,num_steps)*dt #time variable


#-----------------------Integration------------------------#

IC = perturbation(IC0,eps) #perturbed initial conditions

#Initializing arrays

#The following are the solution for each time step,
#for each variable and for each IC
sol_1 = np.zeros((num_steps , 3, len(eps))) 
#chaotic solution 
sol_2 = np.zeros((num_steps , 3, len(eps)))
#non-chaotic solution 


#Integrating

#Solutions for each value of r and for each IC

for i in range(len(eps)):
    
    sol_1[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r1)) 
    #chaotic solution
    sol_2[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r2))
    #non-chaotic solution

    

#-------------------------Analysis-------------------------#

#Initializing arrays

delta_x = np.zeros((num_steps, 2)) 
#The difference was performed only between the solution of the unperturbed 
#case and the one of the first perturbed case, as a preliminary analysis.
#The difference was calculated for both chaotic and non-chaotic solution.

error = np.zeros((num_steps, len(eps))) 
#The RMSE was calculated for each perturbed case with r = 28. 
#The same for the prediction time.

pred_time = np.zeros(3)                 
 

#Calculating

delta_x[:,0] = difference(sol_1[:,:,0], sol_1[:,:,1]) 
delta_x[:,1] = difference(sol_2[:,:,0], sol_2[:,:,1])


for i in range(1,len(eps)): 
    
    error[:,i] = RMSE(sol_1[:,:,0], sol_1[:,:,i])
    
    

pred_time = prediction(error, num_steps, dt, eps)


#------------------------Plots & Tables--------------------------#

#plotting both chaotic and non-chaotic solution for
#the unpertubed case in the x,z plane

xzgraph(sol_1[:,:,0],r1) 
xzgraph(sol_2[:,:,0],r2) 

#3D-plotting both chaotic and non-chaotic solution for
#the unpertubed case 

#plot_3dsolution(sol_1[:,:,0],r1)
#plot_3dsolution(sol_2[:,:,0],r2)

plot_animation(sol_1[:,:,0],r1)
#plot_animation(sol_2[:,:,0],r2)

plot_difference(delta_x[:,0],t, r1) 
plot_difference(delta_x[:,1],t, r2)

for i in range(1,len(eps)): 
   
    plot_rmse(error[:,i],t, r1, eps[i], pred_time[i-1])


#creating a table with the values of the perturbation and
# of the corresponding prediction times

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
        
error1 = np.zeros((num_steps, len(eps)))
error2 = np.ones((num_steps, len(eps)))

def test_pred_time():
    
        time = prediction(error1, num_steps, dt, eps)
        
        assert np.all(time == 0.)
        
def test_pred_time1():
    
        time = prediction(error2, num_steps, dt, eps)
        
        assert np.all(time == 0.)
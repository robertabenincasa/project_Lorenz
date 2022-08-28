# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:15:18 2022

@author: robertabenincasa
"""
from lorenz import (lorenz, perturbation, difference,
                    RMSE, prediction, read_parameters)
import numpy as np
import configparser
from scipy.integrate import odeint

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

num_steps1 = config.get('Integration settings', 'num_steps')
num_steps = int(num_steps1)

dt1 = config.get('Integration settings', 'dt')
dt = float(dt1)

IC01 = config.get('Initial condition', 'IC') #initial condition
IC0 = read_parameters(IC01)

eps1 = config.get('Perturbations', 'eps') #perturbations
eps = read_parameters(eps1)
    

t = np.linspace(0,num_steps,num_steps)*dt #time variable

sol = np.zeros((num_steps , 3, len(eps)+1))
#--------------------------------------------------------------#

IC = perturbation(IC0,eps)

for i in range(len(eps)+1):
    
    sol[:,:,i] = odeint(lorenz,IC[i,:],t,args=(sigma,b,r1)) 


def test_valid_IC():
    
    for i in range(3):  
    
        assert IC0[i] == IC[0,i]

def test_valid_IC_1():
    
    for i in range(1,len(eps)+1):
        
        for m in range(2,3):
         
            assert IC0[m] == IC[i,m]
        
 
def test_diff1():
    
    delta_x = difference(sol[:,:,0],sol[:,:,0])
    
    for i in range(num_steps):
        
        assert delta_x[i] == 0.    
 
    
def test_RMSE():
    
    for j in range(1,len(eps)+1):  
                
        assert np.all(RMSE(sol[:,:,0],sol[:,:,j]) >= 0.)
            
            
def test_RMSE1():
    
    rmse = RMSE(sol[:,:,0],sol[:,:,0])
    
    for i in range(num_steps):
        
        assert rmse[i] == 0.
        
error1 = np.zeros((num_steps, len(eps)+1))
error2 = np.ones((num_steps, len(eps)+1))

def test_pred_time():
    
        time = prediction(error1, num_steps, dt, eps)
        
        assert np.all(time == 0.)
        
def test_pred_time1():
    
        time = prediction(error2, num_steps, dt, eps)
        
        assert np.all(time == 0.)
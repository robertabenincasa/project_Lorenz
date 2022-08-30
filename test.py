# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:15:18 2022

@author: robertabenincasa
"""
from lorenz import (lorenz, perturbation, difference,
                    RMSE, prediction, read_parameters)
import numpy as np
import configparser
import hypothesis
from hypothesis import (given, settings, assume)
import hypothesis.strategies as st
import hypothesis.extra.numpy as exnp
from math import isnan
import pytest

config = configparser.ConfigParser()
config.read('config.ini')

dt1 = config.get('Integration settings', 'dt')
dt = float(dt1)

num_steps1 = config.get('Integration settings', 'num_steps')
num_steps = int(num_steps1)

dim_eps1 = config.get('Perturbations', 'dim_eps')
dim_eps = int(dim_eps1)

t = np.linspace(0,num_steps,num_steps)*dt
#--------------------------------------------------------------#



@given(par = st.floats(allow_nan=None, allow_infinity=None),
       par_1 = st.integers(min_value=0),
       par0 = st.characters(whitelist_categories='LMPSZC'))
@settings(max_examples = 100)
def test_read_parameters(par,par_1,par0):
    par1 = str(par_1)
    par2 = str(par)
    string0 =par0+','+par0
    string1 = par2+','+','
    string2 = ','+par1
    string6 = par2+par2
    
    string3 = par2+','+par1+par1
    string4 = par1+par1+','+par2
    string5 = par2 + ','+par2
        
    with pytest.raises(ValueError) as pytest_ve:
        read_parameters(string0)
        assert pytest_ve.type == ValueError
    with pytest.raises(ValueError) as pytest_ve:
        read_parameters(string1)
        assert pytest_ve.type == ValueError
    with pytest.raises(ValueError) as pytest_ve:
        read_parameters(string2)
        assert pytest_ve.type == ValueError
   # with pytest.raises(ValueError) as pytest_ve:
   #     read_parameters(string6)
   #     assert pytest_ve.type == ValueError
    with pytest.raises(SystemExit) as pytest_se:
        read_parameters(string6)
        assert pytest_se.type == SystemExit
    
    assert type(read_parameters(string3)) == np.ndarray
    assert type(read_parameters(string4)) == np.ndarray
    assert type(read_parameters(string5)) == np.ndarray



@given(state = exnp.arrays(np.dtype(float),(3,num_steps),
       elements = st.floats(min_value = -100,max_value= 100,allow_nan=False,
       allow_infinity=False)), sigma = st.floats(min_value = 0,max_value= 30,
       allow_nan=None,allow_infinity=None),
       b = st.floats(min_value = 0,max_value= 10,allow_nan=None, 
       allow_infinity=None),r = st.floats(min_value = 0,max_value= 30,
       allow_nan=None,allow_infinity=None)
       )
@settings(max_examples = 100)  
def test_lorenz(state,sigma,b,r):
    for i in range(3):
        
        assert len(lorenz(state,t,sigma,b,r)[i]) == num_steps
    
    
@given(eps = exnp.arrays(np.dtype(float), dim_eps,elements = 
    st.floats(min_value = -10,max_value= 10,allow_nan=False, 
    allow_infinity=False)),
    IC0 = exnp.arrays(np.dtype(float), 3,
    elements = st.floats(min_value = -100,max_value= 100,allow_nan=False,
    allow_infinity=False)))
@settings(max_examples = 100)
def test_valid_IC(IC0,eps):
   
    IC = perturbation(IC0,eps)
   
    for i in range(3):  
        
        assert IC0[i] == IC[0,i]
        
    for i in range(1,len(eps)+1):
         
        for m in range(2,3):
            
            assert IC0[m] == IC[i,m]   


        
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3),
       elements = st.floats(min_value = -100,max_value= 100,allow_nan=False,
       allow_infinity=False)))
@settings(max_examples = 100)
def test_diff(sol):
    
    delta_x = difference(sol[:,:],sol[:,:])
    
    for i in range(num_steps):
        
        assert delta_x[i] == 0.    
        
        
 
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3,dim_eps+1),
       elements = st.floats(min_value = -100,max_value= 100,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100)    
def test_RMSE(sol):
    sol1 = np.zeros((num_steps,3))
    sol2 = np.zeros((num_steps,3))
    
    sol1[:,:] =  sol[:,:,0]
    for j in range(1,dim_eps+1):  
               
        sol2[:,:] = sol[:,:,j]
        assert np.all(RMSE(sol1[:,:],sol2[:,:]) >= 0.)
            
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3),
        elements = st.floats(min_value = -100,max_value= 100,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100)           
def test_RMSE1(sol):
    
    rmse = RMSE(sol[:,:],sol[:,:])
    
    for i in range(num_steps):
        
        assert rmse[i] == 0.
        

@given(eps = exnp.arrays(np.dtype(float), dim_eps,
        elements = st.floats(min_value = -10,max_value= 10,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100)   
def test_pred_time(eps):
    
        error = np.zeros((num_steps, dim_eps+1))
        time = prediction(error, num_steps, dt, eps)
        
        assert np.all(time == 0.)
 
        
@given(eps = exnp.arrays(np.dtype(float), dim_eps,
        elements = st.floats(min_value = -10,max_value= 10,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100) 
def test_pred_time1(eps):
    
        error = np.ones((num_steps, dim_eps+1))
        time = prediction(error, num_steps, dt, eps)
        
        assert np.all(time == 0.)
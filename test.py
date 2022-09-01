# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:15:18 2022

@author: robertabenincasa
"""
from lorenz import (lorenz, perturbation, difference,
                    RMSE, prediction, read_parameters)
import numpy as np
import configparser
from hypothesis import (given, settings)
import hypothesis.strategies as st
import hypothesis.extra.numpy as exnp
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
    """ This function tests that the read_parameters function used to read 
    the parameters from the configuration file works properly.
    In particular, it verifies that an error would be raised except for the
    case where the the argument is a string made of numbers separeted by 
    commas.
    
    GIVEN: a string of floats, a string of integers and a string of 
           characters
    WHEN: I apply the read_parameters function to some combinations of them
          defined as string0 to string6.
    THEN: The following properties are satisfied:
        string0: tests that a string made of non numerical characters 
        separated by commas would raise a ValueError.
        string1,2 and 3: test that a string made of floats not properly 
        separated by commas would raise a ValueError for the first 2 and a 
        SystemExit error for the last.
        string 4 and 5 test that applying the read_parameters function to 
        properly written strings would return a valid result, i.e. 
        a np.ndarray.
    """
    
    par1 = str(par_1)
    par2 = str(par)
    string0 =par0+','+par0
    string1 = par2+','+','
    string2 = ','+par1
    string3 = par2+par2
    
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
    with pytest.raises(SystemExit) as pytest_se:
        read_parameters(string3)
        assert pytest_se.type == SystemExit
    

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
    """ This function tests that the lorenz function returns a valid 
    result.
    
    GIVEN: the state vector and the parameters of the system
    WHEN: the lorenz function is applied 
    THEN: the time dimension of the resulting array is correct
   """
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
    
   """ This function tests the perturbation function generates valid ICs.
   
       GIVEN: the original IC and the perturbation array
       WHEN: I apply the perturbation function using those as arguments
       THEN: I verify that the resulting ICs matrix preserves in its first row
       the original IC and that the perturbation is applied only on the x-axis.
       
       """
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
    """ This function tests that difference between two identical trajectory
    is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the difference function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    delta_x = difference(sol[:,:],sol[:,:])
    
    for i in range(num_steps):
        
        assert delta_x[i] == 0.    
        
        
 
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3,dim_eps+1),
       elements = st.floats(min_value = -100,max_value= 100,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100)    
def test_RMSE(sol):
    """ This function tests that the RMSE is a positive quantity for each 
    time step.
        
        GIVEN: a solution for several perturbations
        WHEN: I apply the RMSE function using 2 solutions as arguments
        THEN: I expect to obtain a quantity that is positive at every time
        
    """
    
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
    """ This function tests that RMSE between two identical trajectory
    is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the RMSE function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    
    rmse = RMSE(sol[:,:],sol[:,:])
    
    for i in range(num_steps):
        
        assert rmse[i] == 0.
        

@given(eps = exnp.arrays(np.dtype(float), dim_eps,
        elements = st.floats(min_value = -10,max_value= 10,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100)   
def test_pred_time(eps):
    """ This function tests that, given a RMSE that is equal to zero at 
    every time step, the predictability time is equal to zero too.
    
        GIVEN: a perturbation and a RMSE identically equal to zero
        WHEN: I apply the prediction function with the given efinition of 
        predictability time
        THEN: I expect to obtain zero, i.e. the RMSE never becomes greater
        than 0.5.
        """
    
    error = np.zeros((num_steps, dim_eps+1))
    time = prediction(error, num_steps, dt, eps)
        
    assert np.all(time == 0.)
 
        
@given(eps = exnp.arrays(np.dtype(float), dim_eps,
        elements = st.floats(min_value = -10,max_value= 10,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100) 
def test_pred_time1(eps):
    """This function tests that, given a RMSE that is equal to one at 
    every time step, the predictability time is equal to zero.
        
        GIVEN:a perturbation and a RMSE identically equal to one
        WHEN: I apply the prediction function with the given efinition of 
        predictability time
        THEN: I expect to obtain zero, i.e. the RMSE is always greater
        than 0.5.
    
    """
    
    error = np.ones((num_steps, dim_eps+1))
    time = prediction(error, num_steps, dt, eps)
    
    assert np.all(time == 0.)
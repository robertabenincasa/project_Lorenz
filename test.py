# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:15:18 2022

@author: robertabenincasa
"""
from lorenz import (lorenz, perturbation, difference,
                    RMSE, prediction, read_parameters, ensemble)
import numpy as np
import configparser
from hypothesis import (given, settings)
import hypothesis.strategies as st
import hypothesis.extra.numpy as exnp
import pytest

config = configparser.ConfigParser()
config.read('config.ini')

#-------------------------------------------------------------#

dt1 = config.get('Integration settings', 'dt')
dt = float(dt1)

num_steps = 3000 #due to the analytical features of the Lorenz system, the
#tests are performed up to a smaller number of tiumestep than the one used in 
#the integration, since it would have been uninformative.

sigma1 = config.get('Parameters', 'sigma')
sigma = float(sigma1)

b1 = config.get('Parameters', 'b')
b = float(b1)

r_1 = config.get('Parameters', 'r1') #chaotic solution
r1 = float(r_1)

r_2 =  config.get('Parameters', 'r2') #non-chaotic solution
r2 = float(r_2)

dim_eps1 = config.get('Perturbations', 'dim_eps')
dim_eps = int(dim_eps1)

N1 = config.get('Integration settings', 'N')
N = int(N1)

IC01 = config.get('Initial condition', 'IC') #initial condition
IC0 = read_parameters(IC01)

t = np.linspace(0,num_steps,num_steps)*dt

#max_value and min_value have been set to 50 and -50 respectively for the 
#elements of the trajectories since the attractor is surely confined to this region
#for the ICs given in the configuration. Similarly, the perturbation
#was chosen to be picked from the range (-1.1, 1.1).

#--------------------------------------------------------------#



@given(par = st.floats(allow_nan=None, allow_infinity=None),
       par_1 = st.integers(min_value=0),
       par_2 = st.characters(whitelist_categories='LMPSZC'))
@settings(max_examples = 100)
def test_read_parameters_is_working(par,par_1,par_2):
    """ This function tests that the read_parameters function used to read 
    the parameters from the configuration file works properly.
    In particular, it verifies that an error would be raised except for the
    case where the the argument is a string made of numbers separated by 
    commas.
    
    GIVEN: a string of floats, a string of integers and a string of 
           characters
    WHEN: I apply the read_parameters function to some combinations of them
          defined as string0 to string6.
    THEN: The following properties are satisfied:
        
        ->string0: tests that a string made of non numerical characters 
          separated by commas would raise a ValueError.
        ->string1,2 and 3: test that a string made of floats not properly 
          separated by commas would raise a ValueError for the first 2 and a 
          FormatError for the last.
        ->string 4 and 5 test that applying the read_parameters function to 
          properly written strings would return a valid result, i.e. 
          a np.ndarray.
          
    """
    
    string0 = str(par_2)+','+str(par_2)
    string1 = str(par)+','+','
    string2 = ','+str(par_1)
    string3 = str(par)+str(par)
    
    string4 = str(par_1)+str(par_1)+','+str(par)
    string5 = str(par) + ','+str(par)
        
    
    with pytest.raises(ValueError) as pytest_ve:
        read_parameters(string0)
        assert pytest_ve.type == ValueError, ("Separation by commas is"
        "made in the proper way")
    with pytest.raises(ValueError) as pytest_ve:
        read_parameters(string1)
        assert pytest_ve.type == ValueError, ("Separation by commas is"
        "made in the proper way")
    with pytest.raises(ValueError) as pytest_ve:
        read_parameters(string2)
        assert pytest_ve.type == ValueError, ("Separation by commas is"
        "made in the proper way")
    with pytest.raises(ValueError) as pytest_se:
        read_parameters(string3)
        assert pytest_se.type == ValueError, ("Separation by commas is"
        "made in the proper way")
    

    assert type(read_parameters(string4)) == np.ndarray, "Incorrect use"
    assert type(read_parameters(string5)) == np.ndarray, "Incorrect use"




for r in [r1,r2]:
    @given(state = exnp.arrays(np.dtype(float),(3,num_steps),
       elements = st.floats(min_value = -50,max_value= 50, allow_nan=False,
       allow_infinity=False)))
    @settings(max_examples = 100)  
    def test_lorenz_is_valid(state):
        """ This function tests that the lorenz function returns a valid 
        result.
        
        GIVEN: the state vector and the parameters of the system
        WHEN: the lorenz function is applied 
        THEN: the time dimension of the resulting array is correct
        """
        for i in range(3):
        
            assert len(lorenz(state,t,sigma,b,r)[i]) == num_steps, "Invalid result"
    
    
@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False, 
    allow_infinity=False)))
@settings(max_examples = 100)
def test_IC_is_valid(eps):
    
   """ This function tests that the perturbation function generates
       valid ICs.
   
       GIVEN: the original IC and the perturbation array
       WHEN: I apply the perturbation function using those as arguments
       THEN: I verify that the resulting ICs matrix preserves in its first row
       the original IC and that the perturbation is applied only on the x-axis.
       
       """
   IC = perturbation(IC0,eps)
       
   for i in range(3):  
        
       assert IC0[i] == IC[0,i], "Original IC is not preserved in the 0 row"
        
   for i in range(1,len(eps)+1):
         
       for m in range(2,3):
            
           assert IC0[m] == IC[i,m], "Perturbation is not only on the 0 axis"


        
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3),
       elements = st.floats(min_value = -50,max_value= 50,allow_nan=False,
       allow_infinity=False)))
@settings(max_examples = 100)
def test_diff_identical_trajectories(sol):
    """ This function tests that difference between two identical trajectory
    is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the difference function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    delta_x = difference(sol[:,:],sol[:,:])
    
    for i in range(num_steps):
        
        assert delta_x[i] == 0., ("The difference function is not working"
                                  "properly")
        
        
 
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3,N+1),
       elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100)    
def test_RMSE_positive_quantity(sol):
    """ This function tests that the RMSE is a positive quantity for each 
    time step.
        
        GIVEN: a solution for several perturbations
        WHEN: I apply the RMSE function using 2 solutions as arguments
        THEN: I expect to obtain a quantity that is positive at every time
        
    """
    
    sol1 = np.zeros((num_steps,3))
    sol2 = np.zeros((num_steps,3))
    
    sol1[:,:] =  sol[:,:,0]
    for j in range(1,N+1):  
               
        sol2[:,:] = sol[:,:,j]
        assert np.all(RMSE(sol1[:,:],sol2[:,:]) >= 0.), ("The RMSE function"
                            "is not working properly")
            
        
        
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3),
       elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100)           
def test_RMSE_identical_trajectories(sol):
    """ This function tests that RMSE between two identical trajectory
    is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the RMSE function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    
    rmse = RMSE(sol[:,:],sol[:,:])
    
    for i in range(num_steps):
        
        assert rmse[i] == 0., ("The RMSE function"
                            "is not working properly")
        
        
        
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3,1),
        elements = st.floats(min_value = -100,max_value= 100,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100, deadline=None)           
def test_ensemble_mean(sol):
    """ This function tests that the ensemble mean of an ensemble composed of a
    single trajectory is the trajectory itself.
    
        GIVEN: a single trajectory
        WHEN: I apply the ensemble function 
        THEN: I expect to obtain the same trajectory as ensemble mean 
    """
    
    mean = ensemble(sol)[1]
    
    assert np.array_equal(mean, sol[:,:,0], equal_nan=False) == True, ("The ensemble function"+
                           "is not working properly")
          
    
     
@given(sol = exnp.arrays(np.dtype(float),(num_steps,3,1),
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, allow_infinity=False)))
@settings(max_examples = 100, deadline=None)           
def test_ensemble_spread(sol):
    """ This function tests that the ensemble spread of an ensemble composed of a
    single trajectory is zero.
    
        GIVEN: a single trajectory
        WHEN: I apply the ensemble function 
        THEN: I expect to obtain the zero as ensemble spread  
    """
    
    spread = ensemble(sol)[0]
          
    assert np.all(spread == 0.), ("The ensemble function"+
                            "is not working properly")
            

@given(eps = exnp.arrays(np.dtype(float), N,
        elements = st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100)   
def test_pred_time_with_rmse_equal_to_zero(eps):
    """ This function tests that, given a RMSE that is equal to zero at 
    every time step, the predictability time is equal to zero too.
    
        GIVEN: a perturbation and a RMSE identically equal to zero
        WHEN: I apply the prediction function with the given definition of 
        predictability time
        THEN: I expect to obtain zero, i.e. the RMSE never becomes greater
        than 0.5.
        """
    
    error = np.zeros((num_steps, N+1))
    time = prediction(error, num_steps, dt, eps)
        
    assert np.all(time == 0.), ("The prediction function is not "
    "working properly")
 
        
@given(eps = exnp.arrays(np.dtype(float), N,
        elements = st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100) 
def test_pred_time_with_rmse_equal_to_1(eps):
    """This function tests that, given a RMSE that is equal to one at 
    every time step, the predictability time is equal to zero.
        
        GIVEN:a perturbation and a RMSE identically equal to one
        WHEN: I apply the prediction function with the given efinition of 
        predictability time
        THEN: I expect to obtain zero, i.e. the RMSE is always greater
        than 0.5.
    
    """
    
    error = np.ones((num_steps,N+1))
    time = prediction(error, num_steps, dt, eps)
    
    assert np.all(time == 0.), ("The prediction function is not "
    "working properly")
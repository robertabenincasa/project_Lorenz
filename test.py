# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:15:18 2022

@author: robertabenincasa
"""
import numpy as np
import math
from hypothesis import (given, settings)
import hypothesis.strategies as st
import hypothesis.extra.numpy as exnp
from os import path
import pytest
from unittest import mock
import lorenz

#----------------------------PARAMETERS TO BE SET-----------------------------#

default_file_true = 'config.ini'

NUM_STEPS = 3000 #Is not necessary to test for all the timesteps used in the 
#integration because 

dt = 0.005

t = np.linspace(0,NUM_STEPS,NUM_STEPS)*dt

N = 10 #ensemble members


#----------------------------------TESTING------------------------------------#


@given(default_file = st.text(alphabet=st.characters(whitelist_categories=('L')),
                               min_size=1))
@settings(max_examples = 10)
def test_reading_configuration_file_valid_file(default_file):
    
    """ This function tests that when a non-existing file is given as default 
    configuration file a NameError is raised.
    
        GIVEN: the function reading_configuration_file and a default file
        WHEN: the default configuration file does not exist
        THEN: a NameError is raised.
        """
          
    with mock.patch('builtins.input', return_value=default_file):
        
        with pytest.raises(NameError):
            
            if path.exists(default_file) == False:
            
                lorenz.reading_configuration_file(default_file)
            
                assert pytest.raises(NameError).type is NameError
                

@given(default_file = st.text(alphabet=st.characters(whitelist_categories=('L')),
                               min_size=1))
@settings(max_examples = 10)
def test_reading_configuration_file_not_existing(default_file):
    
    """ This function tests that when a non-existing file is given as input
    by the user a NameError is raised.
    
        GIVEN: the function reading_configuration_file
        WHEN: the configuration file given as command line input by the user 
              does not exist
        THEN: a NameError is raised.
        """
          
    with mock.patch('builtins.input', return_value=default_file):
        
        with pytest.raises(NameError):
        
            if path.exists(default_file) == False:
            
                lorenz.reading_configuration_file(default_file_true)
                
                assert pytest.raises(NameError).type is NameError
            
            
            
def test_reading_configuration_file_default():
    
    """ This function tests that when none configuration file is given as 
    command line input by the user the configuration file is set to be the 
    default one.
    
        GIVEN: the function reading_configuration_file
        WHEN: No input is given by the user
        THEN: The default configuration file is adopted.
        """
    
    with mock.patch('builtins.input', return_value=None):
        
        assert default_file_true == lorenz.reading_configuration_file(default_file_true)



@given(state = exnp.arrays(np.dtype(float),(3,NUM_STEPS),
        elements = st.floats(min_value = -50,max_value= 50, allow_nan=False,
        allow_infinity=False)), b = st.floats(min_value = 0, max_value= 10, 
        allow_nan=False,allow_infinity=False), sigma = st.floats(min_value = 0,
        max_value= 20, allow_nan=False, allow_infinity=False), 
        r = st.floats(min_value = 0,max_value= 30, allow_nan=False,
        allow_infinity=False))
@settings(max_examples = 100)  
def test_lorenz_is_correct(state, sigma, b, r):
    """ This function tests that the lorenz function returns the correct Lorenz 
    system.
        
        GIVEN: the state vector and the parameters of the system
        WHEN: the lorenz function is applied 
        THEN: the output of the function is equal to what expected from the 
        theory, i.e. :
            
            x_dot = sigma * (y - x)
            y_dot = r * x - x * z - y 
            z_dot = x * y - b * z
            
        """
    x_dot, y_dot, z_dot = lorenz.lorenz(state,t,sigma,b,r)
    
    assert np.all(x_dot == sigma * (state[1] - state[0]))
    assert np.all(y_dot == r * state[0] - state[0] * state[2] - state[1])
    assert np.all(z_dot == state[0] * state[1] - b * state[2])
    
    
     
    
@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False, 
    allow_infinity=False)), IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
        st.floats(min_value = -20,max_value= 20,allow_nan=False, 
        allow_infinity=False)), which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 100)
def test_original_ic_is_preserved(eps, IC0, which_variable):
    
    """ This function tests that the perturbation function preserves the 
    original IC in the first row.
   
        GIVEN: the original IC and the perturbation array
        WHEN: I apply the perturbation function
        THEN: I verify that the resulting ICs matrix preserves in its first row
        the original IC.
       
        """
    IC = lorenz.perturbation(IC0,eps,which_variable)
       
    assert np.all(IC0[:] == IC[0,:]), "Original IC is not preserved in the 0 row"
        
    
@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False, 
    allow_infinity=False)), IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
        st.floats(min_value = -20,max_value= 20,allow_nan=False, 
        allow_infinity=False)), which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 100)
def test_ic_is_applied_only_on_the_chosen_axis(eps, IC0, which_variable):
    """ This function tests that the perturbation function applies the 
    perturbation on the IC only on the axis identified by the which_variable 
    parameter.
    
        GIVEN: the original IC and the perturbation array
        WHEN: I apply the perturbation function
        THEN: I verify that the perturbation is applied only on the axis
        identified by the which_variable parameter.
    """
    
    variables_idx = [0,1,2]
    variables_idx.remove(which_variable)
    m,n = variables_idx
    IC = lorenz.perturbation(IC0,eps,which_variable)
    
    ones =  np.ones(N+1)
    assert np.all(IC0[m] * ones == IC[:,m]), ('Perturbation is not only' + 
    ' the chosen axis')
    assert np.all(IC0[n] * ones == IC[:,n]), ("Perturbation is not only" + 
    " the chosen axis")


        
@given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100)
def test_difference_identical_trajectories(sol):
    """ This function tests that the difference function between two identical trajectory
    is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the difference function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    delta = lorenz.difference(sol,sol)
        
    assert np.all(delta == 0.), ("The difference function is not working"
                                  "properly")
    
    
@given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False,
        allow_infinity=False)), sol1 = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 100)
def test_difference_is_correct(sol, sol1):
    """ This function tests that the difference function actually performs the 
    difference between the 2 given array.
    
        GIVEN: 2 trajectory
        WHEN: I apply the difference function 
        THEN: I expect to obtain the difference between them at every time step.
    """
    
    assert np.all(lorenz.difference(sol, sol1) == sol - sol1)
        
  
@given(b = st.floats(min_value = 0, max_value= 10, 
        allow_nan=False,allow_infinity=False), sigma = st.floats(min_value = 0,
        max_value= 20, allow_nan=False, allow_infinity=False), 
        r = st.floats(min_value = 0,max_value= 30, allow_nan=False,
        allow_infinity=False),IC = exnp.arrays(np.dtype(float), (N,3) ,elements = 
        st.floats(min_value = -20,max_value= 20,allow_nan=False, 
        allow_infinity=False)))
@settings(max_examples = 100)         
def test_lorenz_integration(IC, sigma, b, r):
    
    set_ = [sigma, b, r]
    
    sol = lorenz.integration_Lorenz_system(lorenz.lorenz,NUM_STEPS, t, IC, set_)
    
    
    
 
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,N+1),
#        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, allow_infinity=False)))
# @settings(max_examples = 100)    
# def test_rmse_positive_quantity(sol):
#     """ This function tests that the RMSE is a positive quantity for each 
#     time step.
        
#         GIVEN: a solution for several perturbations
#         WHEN: I apply the RMSE function using 2 solutions as arguments
#         THEN: I expect to obtain a quantity that is positive at every time
        
#     """
    
#     sol1 = np.zeros((NUM_STEPS,3))
#     sol2 = np.zeros((NUM_STEPS,3))
    
#     sol1[:,:] =  sol[:,:,0]
#     for j in range(1,N+1):  
               
#         sol2[:,:] = sol[:,:,j]
#         assert np.all(RMSE(sol1[:,:],sol2[:,:]) >= 0.), ("The RMSE function"
#                             "is not working properly")
            
        
        
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
#        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, allow_infinity=False)))
# @settings(max_examples = 100)           
# def test_rmse_identical_trajectories(sol):
#     """ This function tests that RMSE between two identical trajectory
#     is equal to zero.
    
#         GIVEN: a trajectory
#         WHEN: I apply the RMSE function using the former for both
#         arguments
#         THEN: I expect to obtain zero at every time 
#     """
    
#     rmse = RMSE(sol[:,:],sol[:,:])
    
#     for i in range(NUM_STEPS):
        
#         assert rmse[i] == 0., ("The RMSE function"
#                             "is not working properly")
        
        
        
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,1),
#         elements = st.floats(min_value = -100,max_value= 100,allow_nan=False, 
#         allow_infinity=False)))
# @settings(max_examples = 100, deadline=None)           
# def test_ensemble_mean(sol):
#     """ This function tests that the ensemble mean of an ensemble composed of a
#     single trajectory is the trajectory itself.
    
#         GIVEN: a single trajectory
#         WHEN: I apply the ensemble function 
#         THEN: I expect to obtain the same trajectory as ensemble mean 
#     """
    
#     mean = ensemble(sol)[1]
    
#     assert np.array_equal(mean, sol[:,:,0], equal_nan=False) is True, ("The ensemble function"+
#                            "is not working properly")
          
    
     
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,1),
#         elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, allow_infinity=False)))
# @settings(max_examples = 100, deadline=None)           
# def test_ensemble_spread(sol):
#     """ This function tests that the ensemble spread of an ensemble composed of a
#     single trajectory is zero.
    
#         GIVEN: a single trajectory
#         WHEN: I apply the ensemble function 
#         THEN: I expect to obtain the zero as ensemble spread  
#     """
    
#     spread = ensemble(sol)[0]
          
#     assert np.all(spread == 0.), ("The ensemble function"+
#                             "is not working properly")
            

# @given(eps = exnp.arrays(np.dtype(float), N,
#         elements = st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False,
#         allow_infinity=False)))
# @settings(max_examples = 100)   
# def test_pred_time_with_rmse_equal_to_zero(eps):
#     """ This function tests that, given a RMSE that is equal to zero at 
#     every time step, the predictability time is equal to zero too.
    
#         GIVEN: a perturbation and a RMSE identically equal to zero
#         WHEN: I apply the prediction function with the given definition of 
#         predictability time
#         THEN: I expect to obtain zero, i.e. the RMSE never becomes greater
#         than 0.5.
#         """
    
#     error = np.zeros((NUM_STEPS, N))
#     time = prediction(error, NUM_STEPS, dt, eps)
        
#     assert np.all(time == 0.), ("The prediction function is not "
#     "working properly")
 
        
# @given(eps = exnp.arrays(np.dtype(float), N,
#         elements = st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False,
#         allow_infinity=False)))
# @settings(max_examples = 100) 
# def test_pred_time_with_rmse_equal_to_1(eps):
#     """This function tests that, given a RMSE that is equal to one at 
#     every time step, the predictability time is equal to zero.
        
#         GIVEN:a perturbation and a RMSE identically equal to one
#         WHEN: I apply the prediction function with the given efinition of 
#         predictability time
#         THEN: I expect to obtain zero, i.e. the RMSE is always greater
#         than 0.5.
    
#     """
    
#     error = np.ones((NUM_STEPS,N))
#     time = prediction(error, NUM_STEPS, dt, eps)
    
#     assert np.all(time == 0.), ("The prediction function is not "
#     "working properly")


# @given(x = exnp.arrays(np.dtype(float), N,
#         elements = st.floats(min_value = 0,max_value= 1.1,allow_nan=False,
#         allow_infinity=False)), b = st.floats(allow_nan=False, 
#                                               allow_infinity=False))
# @settings(max_examples = 100) 
# def test_func_is_working(x,b):
#     """ This function tests that given an angular coefficient equal to zero, 
#     the linear equation should return y = b for every value of x.
    
#     GIVEN: arbitrary values of x and b 
#     WHEN: a = 0
#     THEN: I expect y to be equal to b for every value of x.
    
#     """
#     assert np.all(func(x,0,b) == b), ("The func function is not "
#     "working properly")
    
    
    
# # @given(c = exnp.arrays(np.dtype(float), N,
# #          elements = st.floats(min_value = 1E-10,max_value= 1.1,allow_nan=False,
# #          allow_infinity=False)))
# @given(c = st.lists(elements = st.floats(min_value = 1E-10,max_value= 1.1,
#                      allow_nan=False,allow_infinity=False),
#                      min_size = 5, max_size = N, unique=True))
# @settings(max_examples = 100)    
# def test_fitting_is_working(c):
#     """ This function tests that given y equal to log(x) the fitting function
#     should return an angular coefficient m equal to one.
    
#     GIVEN: the fitting function
#     WHEN: y = log(x) (base 10)
#     THEN: m =1
#     """
#     c = np.array(c)
#     x = np.sort(c)
        
#     y = np.log10(x)
#     angular_coeff = fitting(func,x,y,1.,0.)[1][0]
   
#     assert math.isclose(angular_coeff, 1.,rel_tol = 1E-7) == True  
    
    
    
    
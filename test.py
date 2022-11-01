# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:15:18 2022

@author: robertabenincasa
"""
import numpy as np
import scipy.stats as ss
from hypothesis import (given, settings, example)
import hypothesis.strategies as st
import hypothesis.extra.numpy as exnp
from os import path
import pytest
from unittest import mock
import lorenz
import configparser

#----------------------------PARAMETERS TO BE SET-----------------------------#

default_file_true = 'config.ini'

configuration_file = 'config.ini' #Change it with the configuration file that
# you are using

NUM_STEPS = 3000 #In certain tests is not necessary to use all the time steps 
#used in the actual integration. That is because the testing is not done on the
# Lorenz system, but on the specific functions defined in lorenz.py .

dt = 0.005

t = np.linspace(0,NUM_STEPS,NUM_STEPS)*dt

N, N1 = 10, 100 #ensemble members

IC_0 = np.array([9, 10, 20])

sigma = 10.

b = 8./3.
#----------------------------------TESTING------------------------------------#


#---------------------------READING_CONFIGURATION_FILE------------------------#


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
            
               
                

@given(file = st.text(alphabet=st.characters(whitelist_categories=('L')),
                                min_size=1))
@settings(max_examples = 10)
def test_reading_configuration_file_not_existing(file):
    
    """ This function tests that when a non-existing file is given as input
    by the user a NameError is raised.
    
        GIVEN: the function reading_configuration_file
        WHEN: the configuration file given as command line input by the user 
              does not exist
        THEN: a NameError is raised.
        """
          
    with mock.patch('builtins.input', return_value=file):
        
        with pytest.raises(NameError):
        
            if path.exists(file) == False:
            
                lorenz.reading_configuration_file(default_file_true)
                
            
            
            
            
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



def test_configuratio_has_right_params():
    """ This function tests that the given configuration file, that is the one
    specified by the user at the beginning of this code, contains the same
    setions and parameters of the default file: config.ini. Moreover, it is
    checked that if it is asked for a non existing section or parameter, a 
    NoSectionError and a NoOptionError will be raised, respectively.
    
        GIVEN: a configuration file
        WHEN: I check for the same section and parameters of the default one
        THEN: I expect to raise no errors
        
        GIVEN: a configuration file
        WHEN: I check for a non existing section and parameters 
        THEN: I expect to raise a NoSectionError or a NoOptionError.
    """
    config = configparser.ConfigParser()
    config.read(configuration_file)
    
    try:
        
        config.get('Paths to files', 'path_data')
        config.get('Paths to files', 'path_plots')
        config.get('Parameters', 'sigma')
        config.get('Parameters', 'b')
        config.get('Parameters', 'r1')
        config.get('Parameters', 'r2')
        config.get('Integration settings', 'num_steps')
        config.get('Integration settings', 'dt')
        config.get('Integration settings', 'N')
        config.get('Initial condition', 'IC')
        config.get('Perturbations', 'which_variable')
        config.get('Perturbations', 'eps')
        config.get('Integration settings', 'Random seed')
        config.get('Analysis', 'Threshold')
        config.get('Plotting','which_eps_for_difference')
        config.get('Plotting','which_eps_for_animation')
        config.get('Plotting','which_eps_for_difference')
        
    except KeyError:        
        
        pytest.fails(KeyError)
        
    with pytest.raises(configparser.NoOptionError):
        
        config.get('Parameters', 'not existing parameter')
        
    with pytest.raises(configparser.NoSectionError):
        
        config.get('not existing section','sigma')
        
        
#------------------------------------LORENZ-----------------------------------#


@given(state = exnp.arrays(np.dtype(float),(3,NUM_STEPS),
        elements = st.floats(allow_nan = False, allow_infinity = False)), 
        b = st.floats(allow_nan = False, allow_infinity = False), 
        sigma = st.floats(allow_nan = False, allow_infinity = False), 
        r = st.floats(allow_nan = False, allow_infinity = False))
@settings(max_examples = 10)  
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
    

    

@given(b = st.floats(min_value = 0.001, max_value= 10, 
        allow_nan=False), sigma = st.floats(min_value = 0.001,
        max_value= 20, allow_nan=False), 
        r1 = st.floats(min_value = 0., max_value= 1., allow_nan=False,
        exclude_min = True, exclude_max = True),
        r2  = st.floats(min_value = 1., max_value= 24., allow_nan=False,
        exclude_min = True, exclude_max = True))
@settings(max_examples = 10)    
def test_critical_points(b, sigma, r1, r2):
    
    """ This function tests that the Lorenz system defined in the lorenz 
    function satisfies some of the properties of the original one, i.e. the
    existence of specific critical points. In particular, it tests that the 
    time derivative of each variable of the system is zero for the following
    points and for the specifed value of r:
        
        -> 0 < r < 1 : [0,0,0]
        -> 1 < r < 24,..: [+/-np.sqrt(b*(r-1)),+/-np.sqrt(b*(r-1)), r-1]
        
        GIVEN: the time derivative given by the lorenz function
        WHEN: I consider the critical points of the real system for different
        values of the parameter r
        THEN: I expect to obtain a zero derivative
        
        """
    
    state_1, t = np.array([0,0,0]), dt
    
    zeros = np.zeros(3)
    
    assert np.allclose(lorenz.lorenz(state_1,t,sigma,b,r1), zeros, 1E-7)  == True
    
    state_2 = np.array([np.sqrt(b*(r2-1)),np.sqrt(b*(r2-1)), r2-1])
    
    assert np.allclose(lorenz.lorenz(state_2,t,sigma,b,r2), zeros, 1E-7) == True
    
    state_3 = np.array([-np.sqrt(b*(r2-1)),-np.sqrt(b*(r2-1)), r2-1])
    
    assert np.allclose(lorenz.lorenz(state_3,t,sigma,b,r2),zeros , 1E-7) == True
    

    
#--------------------------------PERTURBATION---------------------------------#
     
    
@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False)), 
    IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
    st.floats(min_value = -20,max_value= 20,allow_nan=False)), 
    which_variable = st.integers(min_value = 0, max_value= 2))
@settings(max_examples = 10)
def test_original_ic_is_preserved(eps, IC0, which_variable):
    
    """ This function tests that the perturbation function preserves the 
    original IC in the first row.
   
        GIVEN: the original IC and the perturbation array
        WHEN: I apply the perturbation function
        THEN: I verify that the resulting ICs matrix preserves in its first row
        the original IC.
       
        """
    IC = lorenz.perturbation(IC0,eps,which_variable)
       
    assert np.all(IC0[:] == IC[0,:])
        
    
@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False)), 
    IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
    st.floats(min_value = -20,max_value= 20,allow_nan=False)), 
    which_variable = st.integers(min_value = 0, max_value= 2))
@settings(max_examples = 10)
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
    assert np.all(IC0[m] * ones == IC[:,m])
    
    assert np.all(IC0[n] * ones == IC[:,n])


@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(allow_nan = False, allow_infinity = False)), IC0 = 
    exnp.arrays(np.dtype(float), 3 ,elements = 
    st.floats()), which_variable = st.integers(min_value = 0, max_value = 2))
@settings(max_examples = 10)
@example(eps = np.ones(N), IC0 = np.ones(3)*float('nan'), which_variable = 0)
@example(eps = np.ones(N), IC0 = np.ones(3)*float('inf'), which_variable = 0)
def test_perturbation_exceptions_ic(eps, IC0, which_variable):
    
    """ This function tests that the perturbation function raises the expected
    exceptions. In particular, it tests that it raises a ValueError when 
    the chosen initial condition is infinite or a nan and that an IndexError is
    raised when the index is out of bounds for the number of variables.
   
        GIVEN: the arguments of the perturbation function
        WHEN: I apply the perturbation function
        THEN: I verify that a ValueError and IndexError are raised when 
        expected.
       
        """


    if np.any(np.isnan(IC0)) == True or np.any(np.isinf(IC0)) == True:
        
        with pytest.raises(ValueError):
            
            lorenz.perturbation(IC0, eps, which_variable)
            
@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats(allow_nan = False, allow_infinity = False)), IC0 = 
    exnp.arrays(np.dtype(float), 3 ,elements = 
    st.floats(allow_nan = False, allow_infinity = False)), which_variable = st.integers())
@settings(max_examples = 10)
def test_perturbation_exceptions_index_error(eps, IC0, which_variable):
    
    """ This function tests that the perturbation function raises the expected
    exceptions. In particular, it tests that it raises a ValueError when 
    the chosen initial condition is infinite or a nan and that an IndexError is
    raised when the index is out of bounds for the number of variables.
   
        GIVEN: the arguments of the perturbation function
        WHEN: I apply the perturbation function
        THEN: I verify that a ValueError and IndexError are raised when 
        expected.
       
        """            
     
    if which_variable >= 3:
        
        with pytest.raises(IndexError):
                
            lorenz.perturbation(IC0, eps, which_variable)
            

@given(eps = exnp.arrays(np.dtype(float), N ,elements = 
    st.floats()), IC0 = 
    exnp.arrays(np.dtype(float), 3 ,elements = 
    st.floats(allow_nan = False, allow_infinity = False)), 
    which_variable = st.integers(min_value = 0, max_value = 2))
@settings(max_examples = 10)
@example(eps = np.ones(N)*float('nan'), IC0 = np.ones(3), which_variable = 0)
@example(eps = np.ones(N)*float('inf'), IC0 = np.ones(3), which_variable = 0)
def test_perturbation_exceptions_eps(eps, IC0, which_variable):
    
    """ This function tests that the perturbation function raises the expected
    exceptions. In particular, it tests that it raises a ValueError when 
    the applied perturbation is infinite or a nan.
   
        GIVEN: the arguments of the perturbation function
        WHEN: I apply the perturbation function
        THEN: I verify that a ValueError is raised when 
        expected.
       
        """


    if np.any(np.isnan(eps)) == True or np.any(np.isinf(eps)) == True:
        
        with pytest.raises(ValueError):
            
            lorenz.perturbation(IC0, eps, which_variable)
            
    
            


#--------------------------------DIFFERENCE-----------------------------------#

        
@given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False)))
@settings(max_examples = 10)
def test_difference_identical_trajectories(sol):
    """ This function tests that the difference function between two identical 
    trajectory is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the difference function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    delta = lorenz.difference(sol,sol)
        
    assert np.all(delta == 0.)
    
    
@given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50,max_value= 50)),
        sol1 = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50,max_value= 50)))
@settings(max_examples = 10)
def test_difference_is_correct(sol, sol1):
    """ This function tests that the difference function actually performs the 
    difference between the 2 given array.
    
        GIVEN: 2 trajectory
        WHEN: I apply the difference function 
        THEN: I expect to obtain the difference between them at every time step.
    """
    assert np.allclose(lorenz.difference(sol, sol1),np.subtract(sol,sol1),
                        rtol = 1E-5,atol = 1E-7)
    zeros = np.zeros(NUM_STEPS)
    assert np.allclose(lorenz.difference(sol, zeros), sol, rtol = 1E-5, atol = 1E-7)
    assert np.allclose(lorenz.difference(zeros, sol), -sol, rtol = 1E-5, atol = 1E-7)
    
        
@given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50, max_value= 50)), 
      sol1 = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50, max_value= 50)))
@settings(max_examples = 10)  
def test_difference_antisymmetry(sol, sol1):
    """ This function tests that the difference function is antisymmetric.
    
        GIVEN: 2 solution
        WHEN: I compare the difference between the first and the second with the
            one between the second and the first
        THEN: I should obtain the same result, but with opposite sign.
        
    """
    
    assert np.all(lorenz.difference(sol, sol1) == -lorenz.difference(sol1, sol))

@given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
        elements = st.floats(min_value = -50, max_value= 50)), 
      sol1 = exnp.arrays(np.dtype(float),NUM_STEPS-1,
        elements = st.floats(min_value = -50, max_value= 50)))
@settings(max_examples = 10) 
def difference_raises_value_error(sol, sol1):
    
  """ This function tests that if the 2 given arrays have different dimensions
  the function difference raises a ValueError.

      GIVEN:2 arrays with different dimensions
      WHEN: I apply the difference function using them as arguments
      THEN: I should obtain a ValueError.

    """
    
  with pytest.raises(ValueError):
        
      lorenz.difference(sol,sol1)
    

#---------------------------INTEGRATION_LORENZ_SYSTEM-------------------------#


@given(r = st.floats(min_value = 0., max_value= 1., allow_nan=False,
        exclude_min = True, exclude_max = True),
        eps = exnp.arrays(np.dtype(float), N ,elements = 
        st.floats(min_value = 1E-10,max_value= 1.1,allow_nan=False)),  
        which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 10)         
def test_lorenz_integration_zero_is_an_attractor( r, eps, which_variable):
    """ This function tests that the result of the integration satisfies the
    following property of the Lorenz system: zero is an attractor for the system
    for 0 < r < 1.
        
        GIVEN: r = 1  
        WHEN: I call the function integration_Lorenz_system
        THEN: I obtain that the solution for the last time steps is close to zero.
    
    """
    
    NUM_STEPS = 12000
    
    t = np.linspace(0,NUM_STEPS,NUM_STEPS)*dt
    
    set_ = [sigma, b, r]
    
    IC = lorenz.perturbation(IC_0,eps,which_variable)
    
    sol = lorenz.integration_Lorenz_system(lorenz.lorenz,NUM_STEPS, t, IC, set_)
    
    assert np.all(abs(sol[NUM_STEPS-10:NUM_STEPS,:,:]) <= 0.5 )
    


@given(r = st.floats(min_value = 1., max_value= 24., allow_nan=False,
        exclude_min = True, exclude_max = True),
        eps = exnp.arrays(np.dtype(float), N ,elements = 
        st.floats(min_value = 1E-10,max_value= 1.1,allow_nan=False)),  
        which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 10, deadline=None)         
def test_lorenz_integration_critical_points(r, eps, which_variable):
    """ This function tests that the result of the integration satisfies the
    following property of the Lorenz system: zero is an attractor for the system
    for 0 < r < 1.
        
        GIVEN: r = 1  
        WHEN: I call the function integration_Lorenz_system
        THEN: I obtain that the solution for the last time steps is close to zero.
    
    """
    
    NUM_STEPS = 12000
    
    t = np.linspace(0,NUM_STEPS,NUM_STEPS)*dt
    
    set_ = [sigma, b, r]
    
    IC = lorenz.perturbation(IC_0,eps,which_variable)
    
    if sigma > b +1:
    
        sol = lorenz.integration_Lorenz_system(lorenz.lorenz,NUM_STEPS, t, IC, set_)
    
        point_1 = np.array([np.sqrt(b*(r-1)),np.sqrt(b*(r-1)), r-1])*np.ones((3,N+1)).T
        
        point_2 = np.array([-np.sqrt(b*(r-1)),-np.sqrt(b*(r-1)), r-1])*np.ones((3,N+1)).T
        
        if np.all(sol[NUM_STEPS-1,:,:] - point_1.T <= 2):
            
            assert np.all(sol[NUM_STEPS-1,:,:] - point_1.T <= 2) == True
        
        if np.all(sol[NUM_STEPS-1,:,:] - point_2.T <= 2):
        
            assert np.all(sol[NUM_STEPS-1,:,:] - point_2.T <= 2) == True


@given(b = st.floats(), sigma = st.floats(), 
        r = st.floats(),eps = exnp.arrays(np.dtype(float), N ,elements = 
        st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False, 
        allow_infinity=False)), which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 10)         
def test_lorenz_integration_raise_exception(sigma, b, r, eps, which_variable):
    """ This function tests that the if the parameters of the integration are either
    infinite or nan, a warning is raised.
    
        GIVEN: At least one among sigma, b and r equal to infinity or nan
        WHEN: I call the integration_Lorenz_system function
        THEN: I expect to receive a warning.
        
    """
    set_ = [sigma, b, r]
    
    IC = lorenz.perturbation(IC_0,eps,which_variable)
    
    if np.any(np.isnan(set_)) == True or np.any(np.isinf(set_)) == True:
    
        with pytest.warns():
    
            lorenz.integration_Lorenz_system(lorenz.lorenz,NUM_STEPS, t, IC, set_)
    
   
    
 
#------------------------------------RMSE-------------------------------------#   
 

@given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3, N+1),
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False,
        allow_infinity=False)))
@settings(max_examples = 10)    
def test_rmse_positive_quantity(sol):
    """ This function tests that the RMSE is a positive quantity for each 
    time step.
        
        GIVEN: a solution for several perturbations
        WHEN: I apply the RMSE function 
        THEN: I expect to obtain a quantity that is positive at every time
        
    """
    
    
    assert np.all(lorenz.RMSE(sol) >= 0.)
        
        
        
@given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, 
        allow_infinity=False)))
@settings(max_examples = 10)           
def test_rmse_identical_trajectories(sol):
    """ This function tests that RMSE between two identical trajectory
    is equal to zero.
    
        GIVEN: a trajectory
        WHEN: I apply the RMSE function using the former for both
        arguments
        THEN: I expect to obtain zero at every time 
    """
    sol_effective = np.zeros((NUM_STEPS,3,2))
    
    sol_effective[:,:,0] = sol
    sol_effective[:,:,1] = sol
    
    rmse = lorenz.RMSE(sol_effective)
    
    assert np.all(rmse == 0.)
    
    sol_effective[:,:,0] = np.zeros((NUM_STEPS,3))
    sol_effective[:,:,1] = np.ones((NUM_STEPS,3))
    
    rmse = lorenz.RMSE(sol_effective)
    
    assert np.all(rmse == 1. )
    
    
        

#-------------------------GENERATE_RANDOM_PERTURBATION------------------------#


@given(seed = st.integers(min_value = 0, max_value = 2000))
@settings(max_examples=10)
def test_generation_random_numbers(seed):
    """ This function tests that the number generated by generate_random_perturbation
    are uniformily distributed between -0.75 and 0.75, as requested.
    
        GIVEN: a fixed random seed
        WHEN: I apply the generate_random_pertrubation function
        THEN: I find 100 uniformily distributed numbers between -0.75 and 0.75.
    
    """
    
    numbers = lorenz.generate_random_perturbation(seed, N1)
    
    assert np.all(numbers <= 0.75) 
    
    stats, p_value = ss.kstest(numbers, ss.uniform(loc = -0.75, 
                                                    scale = 1.5).cdf, N=N1)
    
    assert p_value > 0.01
 
   
#-----------------------------CALCULATING_L_AND_R-----------------------------#    
    
    
@given(sol1 = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False)), 
        sol2 = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False)), 
        rmse = exnp.arrays(np.dtype(float),(NUM_STEPS,N),
        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False)) )
@settings(max_examples = 10) 
def test_L_is_symmetric(sol1,sol2, rmse):
    """ This function test that the function calculating_L_and_R is 
    symmetric if the first and the second are exchanged.
    
        GIVEN: the true solution and the ensemble mean
        WHEN: I change their order as arguments of the function
        THEN: I should obtain the same result.
    
    """
    assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[0] == 
                  lorenz.calculating_L_and_R(sol2, sol1, rmse)[0])
    
    assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[1] == 
                  lorenz.calculating_L_and_R(sol2, sol1, rmse)[1])
    
    assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[0] >= 0.)
    
    assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[1] >= 0.)
    
    
@given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, 
        allow_infinity=False)), rmse = exnp.arrays(np.dtype(float),(NUM_STEPS,1),
        elements = st.floats(min_value = -50, max_value= 50,allow_nan=False)))
@settings(max_examples = 10)    
def test_L_and_R_are_rmse(sol, rmse):
  """ This functions tests that L and R satisfy some properties of the RMSE. 
  In particular, R is tested to show that if the RMSEs are all equal, R is 
  equal to them. While L is tested to verify that if the true solution and 
  the average one are equal L is zero. A particular case is also shown.

      GIVEN: all equal RMSEs and sol_vaerage = sol_true
      WHEN: I apply calculating_L_and_R
      THEN: I expect to obtain R = rmse and L = 0.

    """
  
  assert np.all(lorenz.calculating_L_and_R(sol, sol, rmse)[1] == 0.)
    
  rmse = np.ones((NUM_STEPS,N)) * rmse
    
  sol1, sol2 = np.ones((NUM_STEPS,3)), np.zeros((NUM_STEPS,3))
    
  assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[1] == 1. )
    
  assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[0] == rmse[:,0] )



#-----------------------------------ENSEMBLE----------------------------------#

        
@given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,1),
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, 
        allow_infinity=False)))
@settings(max_examples = 10, deadline=None)           
def test_ensemble_of_single_member(sol):
    
    """ This function tests that the ensemble mean of an ensemble composed of a
    single trajectory is the trajectory itself and, consequently, that the spread
    is zero.
    
        GIVEN: a trajectory
        WHEN: I apply the ensemble function 
        THEN: I expect to obtain the same trajectory as ensemble mean and zero
              as spread.
    """
    
    spread, mean = lorenz.ensemble(sol)
    
    assert np.array_equal(mean, sol[:,:,0], equal_nan=False) is True
    
    assert np.all(abs(spread) == 0.)

    
    
    
@given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,1),
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, 
        allow_infinity=False)))
@settings(max_examples = 10, deadline=None)           
def test_ensemble_of_equal_members(sol):
    
    """ This function tests that the ensemble mean of N identical members is 
    the member itself and, consequently, that the spread is zero.
    
        GIVEN: an esemble of N identical trajectories
        WHEN: I apply the ensemble function 
        THEN: I expect to obtain the same trajectory as ensemble mean and zero
              as spread.
    """
    
    ones = np.ones((NUM_STEPS,3,N))
    ens_sol_equal = sol * ones
    
    spread, mean = lorenz.ensemble(ens_sol_equal)
    
    assert np.array_equal(mean, sol[:,:,0], equal_nan=False) is True
    
    assert np.all(abs(spread) == 0.)
    
    
@given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,N),
        elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, 
        allow_infinity=False)),idx = st.integers(min_value = 0, max_value = N))
@settings(max_examples = 10, deadline=None)     
def test_ensemble_order_is_not_important(sol,idx):
    
    """ This function tests that if the order of the ensemble members is changed
    the mean and the spread are not modified.
    
        GIVEN:an ensemble of trajectories in 2 different order
        WHEN: I apply the ensemble function to both 
        THEN: I should obtain the same ensemble mean and ensemble spread.
    """
    spread, mean = lorenz.ensemble(sol)
    
    sol_different_order = np.roll(sol,idx)

    spread1, mean1 = lorenz.ensemble(sol_different_order)
    
    assert np.all(mean1 == mean)
    assert np.all(spread1 == spread)


#--------------------------------PREDICTION-----------------------------------#


@given(threshold=st.floats(min_value = 0.,max_value= 1., exclude_min=True))
@settings(max_examples = 10)   
def test_pred_time_with_rmse_equal_to_zero(threshold):
    """ This function tests that, given a RMSE that is equal to zero at 
    every time step, the predictability time is equal to zero too.
    
        GIVEN: a perturbation and a RMSE identically equal to zero
        WHEN: I apply the prediction function with the given definition of 
        predictability time
        THEN: I expect to obtain zero, i.e. the RMSE never becomes greater
        than 0.5.
        """
    
    error = np.zeros((NUM_STEPS, N))
    time = lorenz.prediction(error, dt, threshold)
        
    assert np.all(time == 0.)
    
    error1 = np.zeros(NUM_STEPS)
    time1 = lorenz.prediction(error1, dt, threshold)
        
    assert time1 == 0.
 
        
@given(threshold=st.floats(min_value = 0.,max_value= 1., exclude_min=True,exclude_max=True))
@settings(max_examples = 10) 
def test_pred_time_with_rmse_equal_to_1(threshold):
    """This function tests that, given a RMSE that is equal to one at 
    every time step, the predictability time is equal to zero.
        
        GIVEN:a perturbation and a RMSE identically equal to one
        WHEN: I apply the prediction function with the given efinition of 
        predictability time
        THEN: I expect to obtain zero, i.e. the RMSE is always greater
        than 0.5.
    
    """
    
    error = np.ones((NUM_STEPS,N))
    time = lorenz.prediction(error, dt, threshold)
    
    assert np.all(time == 0.)
    
    error1 = np.ones(NUM_STEPS)
    time1 = lorenz.prediction(error1, dt, threshold)
    
    assert time1 == 0.
    
    
    
@given(threshold=st.floats(min_value = 0.,max_value= 1., exclude_min=True),
        error = exnp.arrays(np.dtype(float),(NUM_STEPS,N),
        elements = st.floats(min_value = 0.,max_value= 50)),
        error1 = exnp.arrays(np.dtype(float),(NUM_STEPS),
        elements = st.floats(min_value = 0.,max_value= 50)))
@settings(max_examples = 10) 
def test_prediction_is_correct(error, error1, threshold):
    
    """This function tests that the predictability time found by the function 
    prediction is correct. In particular, it checks that the rmse after that 
    time step is greater than the threshold. In the case that the predictability
    time is identically equal to zero, it checks if the rmse is always equal to 
    zero too or if it is always greater than the threshold.
    
        GIVEN: the rmse and a certain threshold
        WHEN: I apply the prediction function to find the predictability time
        THEN: I expect to find that the rmse is always greater than the threshold
        after that time step. If the predictability time is equal to zero, I
        verify that the rmse is greater than the threshold at every 
        time step or it is identically equal to zero.
        
    """
    
    time = lorenz.prediction(error, dt, threshold)
    
    if np.all(time != 0.):
    
        for i in range(N):
        
            assert np.all(error[int(time[i]/dt):,i] > threshold)
            
            assert time[i] == np.where(error[i] > threshold)[0][0]*dt            
    
        
    if np.all(error < threshold):
            
        assert np.all(time == 0.)
            
    elif np.all(error > threshold):
            
        assert np.all(time == 0.)
            
    elif np.all(error == threshold):
        
        assert np.all(time == 0.)
        
    elif np.all(error == 0.):
        
        assert np.all(time == 0.)
        
            
    time1 = lorenz.prediction(error1, dt, threshold)
    
    if time1 != 0. :
        
        assert np.all(error1[int(time1/dt):] > threshold)
        
        assert time1 == np.where(error1 > threshold)[0][0]*dt
        
    if np.all(error1 < threshold):
            
        assert time1 == 0.
            
    elif np.all(error1 > threshold):
            
        assert time == 0.
            
    elif np.all(error1 == threshold):
        
        assert time == 0.
        
    elif np.all(error1 == 0.):
        
        assert time == 0.
    
    
        

#-----------------------------------FITTING-----------------------------------#    

    
@given(x = exnp.arrays(np.dtype(float), N,
        elements = st.floats(allow_nan=False, allow_infinity=False)), 
        b = st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples = 10) 
def test_func_with_known_values(x,b):
    """ This function tests that given an angular coefficient equal to zero, 
    the linear equation should return y = b for every value of x. Moreover,
    it also tests that given b = 0 and a = 1, y should be equal to x.
    
    
        GIVEN: arbitrary values of x 
        
            WHEN: a = 0 and b an arbitrary value
            THEN: I expect y to be equal to b for every value of x.
        
            WHEN: a = 1 and b = 0
            THEN: I expect y to be equal to x.
    
    """
    assert np.all(lorenz.func(x,0,b) == b)
    
    assert np.all(lorenz.func(x,1,0) == x)
    
    
@given(b = st.floats())
@settings(max_examples = 10)     
def test_func_with_infinite(b):
    """ This function tests that if x is infinite and a is equal to zero, or 
    viceversa, a RuntimeWarning is raised.
    
        GIVEN: x = infinity and a = 0 or viceversa
        WHEN: I aply the func function
        THEN: I expect to raise a RuntimeWarning
    
    """
    
    a, x = 0., np.ones(N)*np.inf
        
    with pytest.warns(RuntimeWarning):
            
        lorenz.func(x,a,b)
        
    a, x = float('inf'), np.zeros(N)
           
    with pytest.warns(RuntimeWarning):
            
        lorenz.func(x,a,b)
                
    
@given(c = st.lists(elements = st.floats(min_value = 1E-7,max_value= 1.1,
    allow_nan=False,allow_infinity=False),
    min_size = N, max_size = N, unique=True), a =st.floats(min_value = 1.1,
    max_value= 1E7, allow_nan=False), 
    b = st.floats(min_value = 1E-7,max_value= 1.1,
    allow_nan=False) )
@settings(max_examples = 10)        
def test_fitting_linearity_is_exact(c,a,b):
    
    """ This function tests that, given some data that follow a linear 
    equation y = a * x + b  with a and b known, the fitting function find
    exactly a and b as parameters with zero uncertainty.
    
        GIVEN: some data that follow a linear equation y = a * x + b  with a 
        and b known
        WHEN: I call the fitting function with these parameters
        THEN: I find exactly a and b as parameters of the linear equation and
        the uncertainty associated with the fit is zero, i.e. p_low = p_top.
    
    """
    c = np.array(c)
    x = np.sort(c)
        
    y = np.log10(x) * a + b
    
    angular_coeff, q = lorenz.fitting(lorenz.func,x,y,a,b)[1]
    
    p_low, p_top = lorenz.fitting(lorenz.func,x,y,a,b)[2:4]
    
    assert np.isclose(angular_coeff, a,rtol = 1E-7) == True
    
    assert np.isclose(q, b,rtol = 1E-7) == True
    
    if np.any(np.isnan(p_low)) == False and np.any(np.isnan(p_top)) == False:
        
        assert np.allclose(np.ma.getdata(p_low), np.ma.getdata(p_top), rtol = 1E-5, 
                       atol = 1E-7) == True

    
    
@given(x= exnp.arrays(np.dtype(float), N,elements = st.floats()), a =st.floats(),
b = st.floats(), y= exnp.arrays(np.dtype(float), N,elements = st.floats()))
@settings(max_examples = 10)  
def test_fitting_raise_correct_exception(x, a, b, y):
    
    """ This function tests that the fitting function raises a ValueError when 
    one of its arguments is a NaN or infinite.
    
        GIVEN: one of the arguments of the fitting function equal to NaN or 
        infinity
        WHEN: I call the fitting function with these parameters
        THEN: I expect to raise a ValueError.
    """
    
    if np.any(np.isnan(x))== True or np.any(np.isinf(x))== True: 
        
        with pytest.raises(ValueError):
        
            lorenz.fitting(lorenz.func,x,y,a,b)
            
    elif np.any(np.isnan(y))== True or np.any(np.isinf(y))== True: 
        
        with pytest.raises(ValueError):
        
            lorenz.fitting(lorenz.func,x,y,a,b)
            
    elif np.isnan(b)== True or np.isinf(b) == True: 
        
        with pytest.raises(ValueError):
        
            lorenz.fitting(lorenz.func,x,y,a,b)
            
    elif np.isnan(a)== True or np.isinf(a)== True: 
        
        with pytest.raises(ValueError):
        
            lorenz.fitting(lorenz.func,x,y,a,b)
            
@given(x = st.lists(elements = st.floats(min_value = 1E-7,max_value= 1.1,
    allow_nan=False,allow_infinity=False)), a =st.floats(min_value = 1.1,
    max_value= 1E7, allow_nan=False,allow_infinity=False), 
    b = st.floats(min_value = 1E-7,max_value= 1.1,
    allow_nan=False,allow_infinity=False))
@settings(max_examples = 10)
def test_logarithm_in_fitting_function(x, a, b):
    
    """ This function tests that given a value of x less than or equal to zero, 
    when callig the fitting function a RuntimeWarning is raised due to the 
    presence of the logarithm of x.
    
        GIVEN: x such that at least one of its values is negative or equal to 0
        WHEN: I call the fitting function
        THEN: I expect to receive a RuntimeWarning.
    """
    
    x = np.array(x)
    
    y = x + 1 
    
    if np.any(x <= 0.) == True:
    
        with pytest.warns(RuntimeWarning):
            
            lorenz.fitting(lorenz.func,x,y,a,b)
    
    
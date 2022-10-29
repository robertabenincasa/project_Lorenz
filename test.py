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
from scipy.integrate import odeint
#----------------------------PARAMETERS TO BE SET-----------------------------#

default_file_true = 'config.ini'

NUM_STEPS = 3000 #Is not necessary to test for all the timesteps used in the 
#integration because 

dt = 0.005

t = np.linspace(0,NUM_STEPS,NUM_STEPS)*dt

N, N1 = 10, 100 #ensemble members

IC_0 = np.array([9, 10, 20])
#----------------------------------TESTING------------------------------------#


#---------------------------READING_CONFIGURATION_FILE------------------------#


# @given(default_file = st.text(alphabet=st.characters(whitelist_categories=('L')),
#                                min_size=1))
# @settings(max_examples = 10)
# def test_reading_configuration_file_valid_file(default_file):
    
#     """ This function tests that when a non-existing file is given as default 
#     configuration file a NameError is raised.
    
#         GIVEN: the function reading_configuration_file and a default file
#         WHEN: the default configuration file does not exist
#         THEN: a NameError is raised.
#         """
          
#     with mock.patch('builtins.input', return_value=default_file):
        
#         with pytest.raises(NameError):
            
#             if path.exists(default_file) == False:
            
#                 lorenz.reading_configuration_file(default_file)
            
#                 #assert pytest.raises(NameError).type is NameError
                

# @given(default_file = st.text(alphabet=st.characters(whitelist_categories=('L')),
#                                min_size=1))
# @settings(max_examples = 10)
# def test_reading_configuration_file_not_existing(default_file):
    
#     """ This function tests that when a non-existing file is given as input
#     by the user a NameError is raised.
    
#         GIVEN: the function reading_configuration_file
#         WHEN: the configuration file given as command line input by the user 
#               does not exist
#         THEN: a NameError is raised.
#         """
          
#     with mock.patch('builtins.input', return_value=default_file):
        
#         with pytest.raises(NameError):
        
#             if path.exists(default_file) == False:
            
#                 lorenz.reading_configuration_file(default_file_true)
                
#                 #assert pytest.raises(NameError).type is NameError
            
            
            
# def test_reading_configuration_file_default():
    
#     """ This function tests that when none configuration file is given as 
#     command line input by the user the configuration file is set to be the 
#     default one.
    
#         GIVEN: the function reading_configuration_file
#         WHEN: No input is given by the user
#         THEN: The default configuration file is adopted.
#         """
    
#     with mock.patch('builtins.input', return_value=None):
        
#         assert default_file_true == lorenz.reading_configuration_file(default_file_true)


#------------------------------------LORENZ-----------------------------------#


# @given(state = exnp.arrays(np.dtype(float),(3,NUM_STEPS),
#         elements = st.floats()), 
#         b = st.floats(), sigma = st.floats(), 
#         r = st.floats())
# @settings(max_examples = 10)  
# def test_lorenz_is_correct(state, sigma, b, r):
#     """ This function tests that the lorenz function returns the correct Lorenz 
#     system.
        
#         GIVEN: the state vector and the parameters of the system
#         WHEN: the lorenz function is applied 
#         THEN: the output of the function is equal to what expected from the 
#         theory, i.e. :
            
#             x_dot = sigma * (y - x)
#             y_dot = r * x - x * z - y 
#             z_dot = x * y - b * z
            
#         """
#     x_dot, y_dot, z_dot = lorenz.lorenz(state,t,sigma,b,r)
    
#     assert np.all(x_dot == sigma * (state[1] - state[0]))
#     assert np.all(y_dot == r * state[0] - state[0] * state[2] - state[1])
#     assert np.all(z_dot == state[0] * state[1] - b * state[2])
    
    
#--------------------------------PERTURBATION---------------------------------#
     
    
# @given(eps = exnp.arrays(np.dtype(float), N ,elements = 
#     st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False)), 
#     IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
#     st.floats(min_value = -20,max_value= 20,allow_nan=False)), 
#     which_variable = st.integers(min_value = 0, max_value= 2))
# @settings(max_examples = 10)
# def test_original_ic_is_preserved(eps, IC0, which_variable):
    
#     """ This function tests that the perturbation function preserves the 
#     original IC in the first row.
   
#         GIVEN: the original IC and the perturbation array
#         WHEN: I apply the perturbation function
#         THEN: I verify that the resulting ICs matrix preserves in its first row
#         the original IC.
       
#         """
#     IC = lorenz.perturbation(IC0,eps,which_variable)
       
#     assert np.all(IC0[:] == IC[0,:]), "Original IC is not preserved in the 0 row"
        
    
# @given(eps = exnp.arrays(np.dtype(float), N ,elements = 
#     st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False), 
#     IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
#     st.floats(min_value = -20,max_value= 20,allow_nan=False)), 
#     which_variable = st.integers(min_value = 0, max_value= 2))
# @settings(max_examples = 10)
# def test_ic_is_applied_only_on_the_chosen_axis(eps, IC0, which_variable):
#     """ This function tests that the perturbation function applies the 
#     perturbation on the IC only on the axis identified by the which_variable 
#     parameter.
    
#         GIVEN: the original IC and the perturbation array
#         WHEN: I apply the perturbation function
#         THEN: I verify that the perturbation is applied only on the axis
#         identified by the which_variable parameter.
#     """
    
#     variables_idx = [0,1,2]
#     variables_idx.remove(which_variable)
#     m,n = variables_idx
#     IC = lorenz.perturbation(IC0,eps,which_variable)
    
#     ones =  np.ones(N+1)
#     assert np.all(IC0[m] * ones == IC[:,m]), ('Perturbation is not only' + 
#     ' the chosen axis')
#     assert np.all(IC0[n] * ones == IC[:,n]), ("Perturbation is not only" + 
#     " the chosen axis")


# @given(eps = exnp.arrays(np.dtype(float), N ,elements = 
#     st.floats()), IC0 = exnp.arrays(np.dtype(float), 3 ,elements = 
#     st.floats()), which_variable = st.integers())
# @settings(max_examples = 10)
# def test_perturbation_exceptions(eps, IC0, which_variable):
    
#     """ This function tests that the perturbation function raises the expected
#     exceptions. In particular, it tests that it raises a ValueError when 
#     the applied perturbation is infinite or a nan and that an IndexError is
#     raised when the index is out of bounds for the number of variables.
   
#         GIVEN: the arguments of the perturbation function
#         WHEN: I apply the perturbation function
#         THEN: I verify that a ValueError and IndexError are raised when 
#         expected.
       
#         """

#     if np.any(np.isnan(eps)) == True or np.any(np.isinf(eps)) == True:
        
#         with pytest.raises(ValueError):
            
#             lorenz.perturbation(IC0, eps, which_variable)
            
#     if np.any(np.isnan(IC0)) == True or np.any(np.isinf(IC0)) == True:
        
#         with pytest.raises(ValueError):
            
#             lorenz.perturbation(IC0, eps, which_variable)
            
#     if which_variable >= 3:
        
#         with pytest.raises(IndexError):
            
#             lorenz.perturbation(IC0, eps, which_variable)
            
        


#--------------------------------DIFFERENCE-----------------------------------#

        
# @given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
#         elements = st.floats(min_value = -50,max_value= 50,allow_nan=False)))
# @settings(max_examples = 10)
# def test_difference_identical_trajectories(sol):
#     """ This function tests that the difference function between two identical 
#     trajectory is equal to zero.
    
#         GIVEN: a trajectory
#         WHEN: I apply the difference function using the former for both
#         arguments
#         THEN: I expect to obtain zero at every time 
#     """
#     delta = lorenz.difference(sol,sol)
        
#     assert np.all(delta == 0.), ("The difference function is not working"
#                                   "properly")
    
    
# @given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
#         elements = st.floats(min_value = -50,max_value= 50)),
#       sol1 = exnp.arrays(np.dtype(float),NUM_STEPS,
#         elements = st.floats(min_value = -50,max_value= 50)))
# @settings(max_examples = 10)
# def test_difference_is_correct(sol, sol1):
#     """ This function tests that the difference function actually performs the 
#     difference between the 2 given array.
    
#         GIVEN: 2 trajectory
#         WHEN: I apply the difference function 
#         THEN: I expect to obtain the difference between them at every time step.
#     """
    
#     assert np.all(lorenz.difference(sol, sol1) == sol - sol1)
        
# @given(sol = exnp.arrays(np.dtype(float),NUM_STEPS,
#         elements = st.floats(min_value = -50, max_value= 50)), 
#       sol1 = exnp.arrays(np.dtype(float),NUM_STEPS,
#         elements = st.floats(min_value = -50, max_value= 50)))
# @settings(max_examples = 10)  
# def test_difference_antisymmetry(sol, sol1):
#     """ This function tests that the difference function is antisymmetric.
    
#         GIVEN: 2 solution
#         WHEN: I compare the difference between the first and the second with the
#             one between the second and the first
#         THEN: I should obtain the same result, but with opposite sign.
        
#     """
    
#     assert np.all(lorenz.difference(sol, sol1) == -lorenz.difference(sol1, sol))


#---------------------------INTEGRATION_LORENZ_SYSTEM-------------------------#


@given(b = st.floats(min_value = 0.001, max_value= 10, 
        allow_nan=False), sigma = st.floats(min_value = 0.001,
        max_value= 20, allow_nan=False), 
        r = st.floats(min_value = 0., max_value= 1., allow_nan=False,
        exclude_min = True, exclude_max = True),
        eps = exnp.arrays(np.dtype(float), N ,elements = 
        st.floats(min_value = 1E-10,max_value= 1.1,allow_nan=False)),  
        which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 10)         
def test_lorenz_integration_zero_is_an_attractor(sigma, b, r, eps, which_variable):
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
    
    zeros = np.zeros((10,3,N+1))
    
    assert np.all(np.isclose(sol[NUM_STEPS-10:NUM_STEPS,:,:],zeros,rtol=1E-7) == True)
    


@given(b = st.floats(min_value = 0.001, max_value= 10, 
        allow_nan=False), sigma = st.floats(min_value = 0.001,
        max_value= 20, allow_nan=False), 
        r = st.floats(min_value = 1., max_value= 24., allow_nan=False,
        exclude_min = True, exclude_max = True),
        eps = exnp.arrays(np.dtype(float), N ,elements = 
        st.floats(min_value = 1E-10,max_value= 1.1,allow_nan=False)),  
        which_variable = st.integers(min_value = 0,
        max_value= 2))
@settings(max_examples = 10)         
def test_lorenz_integration_critical_points(sigma, b, r, eps, which_variable):
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
    
    if (sigma-b-1) != 0. and r < sigma * (sigma+b+3)/(sigma-b-1):
    
        sol = lorenz.integration_Lorenz_system(lorenz.lorenz,NUM_STEPS, t, IC, set_)
    
        point_1 = [(np.sqrt(b*(r-1))),(np.sqrt(b*(r-1))), r-1]
        
        point_2 = [-(np.sqrt(b*(r-1))),-(np.sqrt(b*(r-1))), r-1]
        
    
        assert (np.all(np.isclose(sol[NUM_STEPS-1,:,0],point_1,rtol=1E-7) == True) or
                np.all(np.isclose(sol[NUM_STEPS-1,:,0],point_2,rtol=1E-7) == True))


# @given(b = st.floats(), sigma = st.floats(), 
#         r = st.floats(),eps = exnp.arrays(np.dtype(float), N ,elements = 
#         st.floats(min_value = -1.1,max_value= 1.1,allow_nan=False, 
#         allow_infinity=False)), which_variable = st.integers(min_value = 0,
#         max_value= 2))
# @settings(max_examples = 10)         
# def test_lorenz_integration(sigma, b, r, eps, which_variable):
    
#     set_ = [sigma, b, r]
    
#     IC = lorenz.perturbation(IC_0,eps,which_variable)
    
#     if np.any(np.isnan(set_)) == True or np.any(np.isinf(set_)) == True:
    
#         with pytest.warns():
    
#             lorenz.integration_Lorenz_system(lorenz.lorenz,NUM_STEPS, t, IC, set_)
    
   
    
 
#------------------------------------RMSE-------------------------------------#   
 

# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3, N+1),
#         elements = st.floats(min_value = -50,max_value= 50,allow_nan=False,
#         allow_infinity=False)))
# @settings(max_examples = 10)    
# def test_rmse_positive_quantity(sol):
#     """ This function tests that the RMSE is a positive quantity for each 
#     time step.
        
#         GIVEN: a solution for several perturbations
#         WHEN: I apply the RMSE function 
#         THEN: I expect to obtain a quantity that is positive at every time
        
#     """
    
    
#     assert np.all(lorenz.RMSE(sol) >= 0.), ("The RMSE function"
#                             "is not working properly")
            
        
        
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
#         elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, 
#         allow_infinity=False)))
# @settings(max_examples = 10)           
# def test_rmse_identical_trajectories(sol):
#     """ This function tests that RMSE between two identical trajectory
#     is equal to zero.
    
#         GIVEN: a trajectory
#         WHEN: I apply the RMSE function using the former for both
#         arguments
#         THEN: I expect to obtain zero at every time 
#     """
#     sol_effective = np.zeros((NUM_STEPS,3,2))
    
#     sol_effective[:,:,0] = sol
#     sol_effective[:,:,1] = sol
    
#     rmse = lorenz.RMSE(sol_effective)
    
#     assert np.all(rmse == 0.), ("The RMSE function"
#                             "is not working properly")
        

#-------------------------GENERATE_RANDOM_PERTURBATION------------------------#


# @given(seed = st.integers(min_value = 0))
# @settings(max_examples=10)
# def test_generation_random_numbers(seed):
#     """ This function tests that the number generated by generate_random_perturbation
#     are uniformily distributed between -0.75 and 0.75, as requested.
    
#         GIVEN: a fixed random seed
#         WHEN: I apply the generate_random_pertrubation function
#         THEN: I find 100 uniformily distributed numbers between -0.75 and 0.75.
    
#     """
    
#     numbers = lorenz.generate_random_perturbation(seed, N1)
    
#     assert np.all(numbers <= 0.75) 
    
#     stats, p_value = ss.kstest(numbers, ss.uniform(loc = -0.75, 
#                                                    scale = 1.5).cdf, N=N1)
    
#     assert p_value > 0.05
 
   
#-----------------------------CALCULATING_L_AND_R-----------------------------#    
    
    
# @given(sol1 = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
#         elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, 
#         allow_infinity=False)), sol2 = exnp.arrays(np.dtype(float),(NUM_STEPS,3),
#         elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, 
#         allow_infinity=False)), rmse = exnp.arrays(np.dtype(float),(NUM_STEPS,N),
#         elements = st.floats(min_value = -50, max_value= 50,allow_nan=False, 
#         allow_infinity=False)) )
# @settings(max_examples = 10) 
# def test_L_is_symmetric(sol1,sol2, rmse):
#     """ This function test that the function calculating_L_and_R is 
#     symmetric if the first and the second are exchanged.
    
#         GIVEN: the true solution and the ensemble mean
#         WHEN: I change their order as arguments of the function
#         THEN: I should obtain the same result.
    
#     """
#     assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[0] == 
#                   lorenz.calculating_L_and_R(sol2, sol1, rmse)[0])
    
#     assert np.all(lorenz.calculating_L_and_R(sol1, sol2, rmse)[1] == 
#                   lorenz.calculating_L_and_R(sol2, sol1, rmse)[1])
    



#-----------------------------------ENSEMBLE----------------------------------#

        
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,1),
#         elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, 
#         allow_infinity=False)))
# @settings(max_examples = 10, deadline=None)           
# def test_ensemble_of_single_member(sol):
    
#     """ This function tests that the ensemble mean of an ensemble composed of a
#     single trajectory is the trajectory itself and, consequently, that the spread
#     is zero.
    
#         GIVEN: a trajectory
#         WHEN: I apply the ensemble function 
#         THEN: I expect to obtain the same trajectory as ensemble mean and zero
#              as spread.
#     """
    
#     mean, spread = lorenz.ensemble(sol)
    
#     assert np.array_equal(mean, sol[:,:,0], equal_nan=False) is True, ("The ensemble function"+
#                             "is not working properly")
    
#     assert np.all(abs(spread) == 0.), ("The ensemble function"+
#                             "is not working properly")

    
    
    
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,1),
#         elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, 
#         allow_infinity=False)))
# @settings(max_examples = 10, deadline=None)           
# def test_ensemble_of_equal_members(sol):
    
#     """ This function tests that the ensemble mean of N identical members is 
#     the member itself and, consequently, that the spread is zero.
    
#         GIVEN: an esemble of N identical trajectories
#         WHEN: I apply the ensemble function 
#         THEN: I expect to obtain the same trajectory as ensemble mean and zero
#              as spread.
#     """
    
#     ones = np.ones((NUM_STEPS,3,N))
#     ens_sol_equal = sol * ones
    
#     mean, spread = lorenz.ensemble(ens_sol_equal)
    
#     assert np.array_equal(mean, sol[:,:,0], equal_nan=False) is True, ("The ensemble function"+
#                             "is not working properly")
#     assert np.all(abs(spread) == 0.), ("The ensemble function"+
#                             "is not working properly")
    
    
# @given(sol = exnp.arrays(np.dtype(float),(NUM_STEPS,3,N),
#         elements = st.floats(min_value = -50,max_value= 50,allow_nan=False, 
#         allow_infinity=False)),idx = st.integers(min_value = 0, max_value = N))
# @settings(max_examples = 10, deadline=None)     
# def test_ensemble_order_is_not_important(sol,idx):
    
#     """ This function tests that if the order of the ensemble members is changed
#     the mean and the spread are not modified.
    
#         GIVEN:an ensemble of trajectories in 2 different order
#         WHEN: I apply the ensemble function to both 
#         THEN: I should obtain the same ensemble mean and ensemble spread.
#     """
#     mean, spread = lorenz.ensemble(sol)
    
#     sol_different_order = np.roll(sol,idx)

#     mean1, spread1 = lorenz.ensemble(sol_different_order)
    
#     assert np.all(mean1 == mean)
#     assert np.all(spread1 == spread)


#--------------------------------PREDICTION-----------------------------------#


# @given(threshold=st.floats(min_value = 0.,max_value= 1., exclude_min=True))
# @settings(max_examples = 10)   
# def test_pred_time_with_rmse_equal_to_zero(threshold):
#     """ This function tests that, given a RMSE that is equal to zero at 
#     every time step, the predictability time is equal to zero too.
    
#         GIVEN: a perturbation and a RMSE identically equal to zero
#         WHEN: I apply the prediction function with the given definition of 
#         predictability time
#         THEN: I expect to obtain zero, i.e. the RMSE never becomes greater
#         than 0.5.
#         """
    
#     error = np.zeros((NUM_STEPS, N))
#     time = lorenz.prediction(error, dt, threshold)
        
#     assert np.all(time == 0.), ("The prediction function is not "
#     "working properly")
    
#     error1 = np.zeros(NUM_STEPS)
#     time1 = lorenz.prediction(error1, dt, threshold)
        
#     assert time1 == 0., ("The prediction function is not "
#     "working properly")
 
        
# @given(threshold=st.floats(min_value = 0.,max_value= 1., exclude_min=True))
# @settings(max_examples = 10) 
# def test_pred_time_with_rmse_equal_to_1(threshold):
#     """This function tests that, given a RMSE that is equal to one at 
#     every time step, the predictability time is equal to zero.
        
#         GIVEN:a perturbation and a RMSE identically equal to one
#         WHEN: I apply the prediction function with the given efinition of 
#         predictability time
#         THEN: I expect to obtain zero, i.e. the RMSE is always greater
#         than 0.5.
    
#     """
    
#     error = np.ones((NUM_STEPS,N))
#     time = lorenz.prediction(error, dt, threshold)
    
#     assert np.all(time == 0.), ("The prediction function is not "
#     "working properly")
    
#     error1 = np.ones(NUM_STEPS)
#     time1 = lorenz.prediction(error1, dt, threshold)
    
#     assert time1 == 0., ("The prediction function is not "
#     "working properly")
    
    
    
# @given(threshold=st.floats(min_value = 0.,max_value= 1., exclude_min=True),
#        error = exnp.arrays(np.dtype(float),(NUM_STEPS,N),
#        elements = st.floats(min_value = -50,max_value= 50)),
#        error1 = exnp.arrays(np.dtype(float),(NUM_STEPS),
#        elements = st.floats(min_value = -50,max_value= 50)))
# @settings(max_examples = 10) 
# def test_prediction_is_correct(error, error1, threshold):
    
#     """This function tests that the predictability time found by the function 
#     prediction is correct. In particular, it checks that the rmse after that 
#     time step is greater than the threshold. In the case that the predictability
#     time is identically equal to zero, it checks if the rmse is always equal to 
#     zero too or if it is always greater than the threshold.
    
#         GIVEN: the rmse and a certain threshold
#         WHEN: I apply the prediction function to find the predictability time
#         THEN: I expect to find that the rmse is always greater than the threshold
#         after that time step. If the predictability time is equal to zero, I
#         verify that the rmse is greater than the threshold at every 
#         time step or it is identically equal to zero.
        
#     """
    
#     time = lorenz.prediction(error, dt, threshold)
    
#     if np.all(time != 0.):
    
#         for i in range(N):
        
#             assert np.all(error[int(time[i]/dt):,i] >= threshold)
    
#     else: 
        
#         assert np.all(error == 0.) or np.all(error >= threshold)
    
#     time1 = lorenz.prediction(error1, dt, threshold)
    
#     if time1 != 0. :
        
#         assert np.all(error1[int(time1/dt):] >= threshold)
    
#     else: 
        
#         assert np.all(error1 == 0.) or np.all(error >= threshold)
    

#-----------------------------------FITTING-----------------------------------#    

    
# @given(x = exnp.arrays(np.dtype(float), N,
#         elements = st.floats(allow_nan=True, allow_infinity=False)), 
#        b = st.floats(allow_nan=True, allow_infinity=False))
# @settings(max_examples = 10) 
# def test_func_with_known_values(x,b):
#     """ This function tests that given an angular coefficient equal to zero, 
#     the linear equation should return y = b for every value of x. Moreover,
#     it also tests that given b = 0 and a = 1, y should be equal to x.
    
    
#         GIVEN: arbitrary values of x 
        
#             WHEN: a = 0 and b an arbitrary value
#             THEN: I expect y to be equal to b for every value of x.
        
#             WHEN: a = 1 and b = 0
#             THEN: I expect y to be equal to x.
    
#     """
#     assert np.all(lorenz.func(x,0,b) == np.ones(N)*b), ("The func function is not "
#     "working properly")
    
#     assert np.all(lorenz.func(x,1,0) == x), ("The func function is not "
#     "working properly")
    
    
# @given(b = st.floats())
# @settings(max_examples = 10)     
# def test_func_with_infinite(b):
#     """ This function tests that if x is infinite and a is equal to zero, or 
#     viceversa, a RuntimeWarning is raised.
    
#         GIVEN: x = infinity and a = 0 or viceversa
#         WHEN: I aply the func function
#         THEN: I expect to raise a RuntimeWarning
    
#     """
    
#     a, x = 0., np.ones(N)*np.inf
        
#     with pytest.warns(RuntimeWarning):
            
#         lorenz.func(x,a,b)
        
#     a, x = float('inf'), np.zeros(N)
           
#     with pytest.warns(RuntimeWarning):
            
#         lorenz.func(x,a,b)
            
# @given(c = st.lists(elements = st.floats(min_value = 1E-7,max_value= 1.1,
#     allow_nan=False),min_size = N, max_size = N, unique=True), 
#     best_guess_1 = st.floats(allow_nan=False, min_value = -5000, max_value = 5000),
#     best_guess_2 = st.floats(allow_nan=False,min_value = -5000, max_value = 5000))
# @settings(max_examples = 10)
# def test_fit_uncertainty(c,best_guess_1,best_guess_2):

#     c = np.array(c)
   
#     x = np.sort(c) 
#     y = np.log10(x) * best_guess_1  + best_guess_2
   
#     fit, popt, p_low, p_top, fit_ = lorenz.fitting(lorenz.func,x,y,best_guess_1+1,
#                                                   best_guess_2+1)      
        
#     if best_guess_1 != 0. :

#         for i in range(N):
       
#             l = []
       
#             for j in range(50):
   
#                 l.append(fit_[j][i])
           
#                 assert np.where(l <= p_low[i])[0].shape[0] <= 4
   
#                 assert np.where(l <= p_top[i])[0].shape[0] <= 49
    
    
    
# @given(c = st.lists(elements = st.floats(min_value = 1E-7,max_value= 1.1,
#     allow_nan=False,allow_infinity=False),
#     min_size = N, max_size = N, unique=True), a =st.floats(min_value = 1.1,
#     max_value= 1E7, allow_nan=False,allow_infinity=False), 
#     b = st.floats(min_value = 1E-7,max_value= 1.1,
#     allow_nan=False,allow_infinity=False) )
# @settings(max_examples = 10)        
# def test_fitting_linearity_is_exact(c,a,b):
    
#     """ This function tests that, given some data that follow a linear 
#     equation y = a * x + b  with a and b known, the fitting function find
#     exactly a and b as parameters with zero uncertainty.
    
#         GIVEN: some data that follow a linear equation y = a * x + b  with a 
#         and b known
#         WHEN: I call the fitting function with these parameters
#         THEN: I find exactly a and b as parameters of the linear equation and
#         the uncertainty associated with the fit is zero, i.e. p_low = p_top.
    
#     """
#     c = np.array(c)
#     x = np.sort(c)
        
#     y = np.log10(x) * a + b
    
#     angular_coeff, q = lorenz.fitting(lorenz.func,x,y,a,b)[1]
    
#     p_low, p_top = lorenz.fitting(lorenz.func,x,y,a,b)[2:4]
    
#     assert np.isclose(angular_coeff, a,rtol = 1E-7) == True
    
#     assert np.isclose(q, b,rtol = 1E-7) == True
    
#     assert np.all(np.isclose(p_low, p_top, rtol = 1E-7)) == True
    
    
# @given(x= exnp.arrays(np.dtype(float), N,elements = st.floats()), a =st.floats(),
# b = st.floats(), y= exnp.arrays(np.dtype(float), N,elements = st.floats()))
# @settings(max_examples = 10)  
# def test_fitting_raise_correct_exception(x, a, b, y):
    
#     """ This function tests that the fitting function raises a ValueError when 
#     one of its arguments is a NaN or infinite.
    
#         GIVEN: one of the arguments of the fitting function equal to NaN or 
#         infinity
#         WHEN: I call the fitting function with these parameters
#         THEN: I expect to raise a ValueError.
#     """
    
#     if np.any(np.isnan(x))== True or np.any(np.isinf(x))== True: 
        
#         with pytest.raises(ValueError):
        
#             lorenz.fitting(lorenz.func,x,y,a,b)
            
#     elif np.any(np.isnan(y))== True or np.any(np.isinf(y))== True: 
        
#         with pytest.raises(ValueError):
        
#             lorenz.fitting(lorenz.func,x,y,a,b)
            
#     elif np.isnan(b)== True or np.isinf(b) == True: 
        
#         with pytest.raises(ValueError):
        
#             lorenz.fitting(lorenz.func,x,y,a,b)
            
#     elif np.isnan(a)== True or np.isinf(a)== True: 
        
#         with pytest.raises(ValueError):
        
#             lorenz.fitting(lorenz.func,x,y,a,b)
            
# @given(x = st.lists(elements = st.floats(min_value = 1E-7,max_value= 1.1,
#     allow_nan=False,allow_infinity=False)), a =st.floats(min_value = 1.1,
#     max_value= 1E7, allow_nan=False,allow_infinity=False), 
#     b = st.floats(min_value = 1E-7,max_value= 1.1,
#     allow_nan=False,allow_infinity=False))
# @settings(max_examples = 10)
# def test_logarithm_in_fitting_function(x, a, b):
    
#     """ This function tests that given a value of x less than or equal to zero, 
#     when callig the fitting function a RuntimeWarning is raised due to the 
#     presence of the logarithm of x.
    
#         GIVEN: x such that at least one of its values is negative or equal to 0
#         WHEN: I call the fitting function
#         THEN: I expect to receive a RuntimeWarning.
#     """
    
#     x = np.array(x)
    
#     y = x + 1 
    
#     if np.any(x <= 0.) == True:
    
#         with pytest.warns(RuntimeWarning):
            
#             lorenz.fitting(lorenz.func,x,y,a,b)
    
    
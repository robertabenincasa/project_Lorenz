# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:07:33 2022

@author: roberta benincasa
"""
from os import path
import numpy as np
from scipy.stats import uniform
from typing import Callable, Union
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import scipy.stats as st
from scipy.stats.mstats import mquantiles




def reading_configuration_file(default_file: str,
                               )->str:
    
    """ This function allows a command line interface with the user in order
    to let them choose the configuration file that they want to use for the 
    simulation.
    If none is given, the default one is used. 
    
    Arguments:
    ----------
        default_file: string
        It is the default configuration file.
        
    Returns:
    --------
        configuration_file: string
        The name of the configuration file chosen by the user, only if it is an
        existing file. Otherwise, the user will be asked to insert the name of 
        an existing file until they provide it.
        
    Note:
    -----
        The user is supposed to insert the entire path of the file if it is not
        in the same folder as the codes.
        
        
        """
   
        
    if path.exists(default_file) == False:        
        
        raise NameError('Invalid default configuration file - This file does '+
                        'not exist')
    
    while True:
        
        print('\n'+'Enter the name of the configuration file of your choice and press' +
          ' ENTER.'+' If none is given, the default one will be used: '+
              str(default_file))
    
        configuration_file = str(input("Name of the file: ") or default_file)
    
        if path.exists(configuration_file) == True:
        
            return configuration_file
    
        else:
            
            raise NameError('\n'+'--------------------->FILE '+ configuration_file +
                  ' DOES NOT EXIST. PLEASE TRY AGAIN<--------------------')
        



def lorenz(
           state_vector: np.ndarray,
           t: np.ndarray,
           sigma: float,
           b: float,
           r: float,
           ) -> list: 
    """" This function returns the time derivative of the 3 variables x, y 
         and z as given by the Lorenz system.
    
         The Lorenz system is a simplified model of atmospheric convection. 
         The 3 variables represent the intensity of the convective flux, the
         temperature difference between the ascending and the descending 
         branch and the deviation from a linear temperature profile at the 
         centre of the cell, respectively. Moreover, it is a canonical 
         example of a system that shows chaotic behaviour for certain values 
         of the parameters.
        
        Arguments:
        ----------
            state_vector : ndarray-like(float) 
            It is the vector formed by the 3 variables x(t), y(t) and z(t) of 
            the Lorenz system.
            
            t: ndarray-like(float)
            Time.
            
            sigma : scalar(float)
            First parameter of the system. It is proportional to the Prandtl 
            number and it is a measure of diffusivity.
            
            b : scalar(float)
            Second parameter of the system. It represents some physical and 
            geometrical properties of the problem.
            
            r : scalar(float)
            Third parameter of the system. It is proportional to the Rayleigh
            number and it gives information about the nature of the flow.
            
        Returns:
        --------
            [x_dot,y_dot,z_dot] : list(float)
            Time derivative of the 3 variables x, y and z as obtained from the
            Lorenz system.
            
        Notes:
        ------
            The canonical choice of parameters to obtain a chaotic solution 
            is sigma = 10 , b = 8/3 and r = 28.
            
    """
    x,y,z=state_vector    
    
    x_dot = sigma * (y - x)
    y_dot = r * x - x * z - y 
    z_dot = x * y - b * z
    
    return [x_dot,y_dot,z_dot]


def perturbation(
                 init_cond: np.ndarray,
                 eps: np.ndarray,
                 which_variable: int,
                 ) -> np.ndarray:
    
    """ This function adds a perturbation to the selected component of the 
    initial condition of the simulation.
    
        Arguments:
        ----------
            init_cond : array-like(float)
            Unperturbed initial values of the 3 variables of the system.
            
            eps : array-like(float)
            Three different values used to perturb the initial condition.
            
            which_variable: integer
            Variable whose IC is to be pertubed. Possible values are: 0, 1 and 2.
        
        Returns:
        --------
            IC : ndarray-like(float)
            Matrix where each rows represents a set of initial conditions. The 
            first row are the unperturbed ones.
            
        Notes:
        ------
            The number of perturbed ICs depends on the number of 
            perturbations. 
    
    
    """
    
    IC = np.ones((eps.shape[0]+1,3))
    
    IC = IC * init_cond
    
    IC[1:,which_variable] = IC[1:,which_variable]+eps
    
    return IC

def integration_Lorenz_system(lorenz: Callable,
                              num_steps: int,
                              t: np.ndarray,
                              IC: np.ndarray,
                              set_parameters: tuple,
                              ) -> np.ndarray:
    """ This function performs the integration of the Lorenz system, defined in
    the function lorenz, using the scipy module odeint.
    
    Arguments:
    ----------
        lorenz: Callable
        It is the function defined above that defines the lorenz system.
        It is necessary to perform the integration using the odeint module.
        
        num_steps: integer
        Number of steps for the integration.
        
        t: np.ndarray(floats)
        Time.
        
        IC: np.ndarray
        Vector with the initial condition for each value of the perturbation.
        
        set_parameters: tuple(floats)
        Chosen set of the values of the 3 parameters of the system: sigma, b 
        and r.
        
    Returns:
    --------
        sol: np.ndarray(floats)
        Solution of the integration of the Lorenz system with the given set of
        parameters. It is a 3D array: the first dimension is the time, the 
        second the variable (x, y or z) and the last is relative to which 
        perturbation was applied to the system. 
        
    Note:
    -----
        sol[:,:,0] represents the unperturbed solution.
        
        """
    
    sol = np.zeros((num_steps, 3, IC.shape[0])) 
    
    sigma, b, r = set_parameters
    
    sol[:,:,0] = odeint(lorenz,IC[0,:],t,args=(sigma,b,r)) 
    
    for i in range(IC.shape[0]-1):
        
        sol[:,:,i+1] = odeint(lorenz,IC[i+1,:],t,args=(sigma,b,r)) 
        
    return sol
    


def difference(
               sol1: np.ndarray,
               sol2: np.ndarray,
               ) -> np.ndarray:
    
    """ This function performs the difference between the x-components of 
        2 trajectories of the system.
        
        It is a preliminary method to obtain a measure of the divergence of 
        2 trajectories of the same system given a slightly difference in 
        the initial conditions adopted for the integration. 
        
            Arguments:
            ----------
                sol1 : ndarray-like(float)
                First solution, the unperturbed one.
                
                sol2 : ndarray-like(float)
                Second solution, the perturbed one.
                
            Returns:
            --------
                delta_x : array-like(float)
                Difference between the x-components of the 2 solutions.
                
    """
    
    delta_x = sol1 - sol2
    
    return delta_x 



def RMSE(
        sol: np.ndarray,
        ) -> np.ndarray:
    
    """This function performs the calculation of the root mean square error 
       of the solution obtained from the perturbed ICs with respect to the 
       unperturbed one.
       
       It is a measure of the divergence of 2 trajectories of the same
       system given a slightly difference in the initial conditions adopted 
       for the integration.
       
       
           Arguments:
           ----------
               sol : ndarray-like(float)
               Integrated trajectory of dimension: (number of steps, number of
               variables, number of perturbations).
               
           Returns:
           --------
               rmse : ndarray-like(float)
               Root Mean Square Error as a function of time and of the applied
               perturbation. It is defined as the square root of the sum of the
               squared differences between the corresponding components of the
               2 trajectories.
               
        
    """
    
    num_steps = sol.shape[0]
    number_of_perturbations = sol.shape[2]-1
    
    rmse = np.zeros((num_steps, number_of_perturbations))
    
    for i in range(number_of_perturbations):
        
        error = np.square(np.subtract(sol[:,:,0],sol[:,:,i+1]))
        rmse[:,i] = np.sqrt(np.mean(error, axis = 1))
                     
    return rmse



def generate_random_perturbation(
                                 random_seed: int,
                                 N: int,
                                 ) -> np.ndarray:
    
    """This function returns an array of N random numbers in the range 
       [-0.75, 0.75].
    
       Arguments:
       ----------
            random_seed: integer
            Fixed value of the random seed.
        
            N: integer
            Number of random numbers to be generated.
        
        Returns:
        --------
            eps: np.ndarray(floats)
            Array of dimension N containing N random numbers in the range
            [-0.75, 0.75).
            
            random: np.ndarray(floats)
            Array of dimension N containing N random numbers in the range
            [0, 1.).
                
    
    
    """
    
    rng = np.random.default_rng(random_seed)

    eps = uniform.rvs(loc = -0.75, scale = 1.5,size=N,random_state=rng)
    
    return eps


def calculating_L_and_R(
                        sol_true: np.ndarray,
                        sol_average: np.ndarray,
                        rmse: np.ndarray,
                        ) -> tuple:
    
    """ This function calculates the mean of the RMSE of each members of the 
        ensemble (R) and the RMSE of the ensemble mean (L).
        
        
            Arguments:
            ----------
                sol_true: np.ndarray(floats)
                'True' solution of the system, i.e. unperturbed one.
                
                sol_average: np.ndarray(floats)
                Ensemble mean.
                
                rmse: np.ndarray(floats)
                RMSEs of each member of the ensemble with respect to the true
                solution.
                
            Returns:
            --------
                R: np.ndarray(floats)
                Mean of the RMSEs of each members of the ensemble, as a 
                function of time.
                
                L: np.ndarray(floats)
                RMSE of the ensemble mean, as a function of time.


    """
    
    
    R = np.mean(rmse, 1)
    
    L = np.square(np.subtract(sol_true,sol_average))
    L = np.sqrt(np.mean(L, axis = 1))
    
    return R, L
    
    

def ensemble(sol: np.ndarray,
                    ) -> tuple:

    """This function performs the calculation of the ensemble mean and of the 
       ensemble spread.
        
           Arguments:
           ----------
               num_steps : int
               Number of timesteps for the integration.
               
               sol: ndarray-like(float)
               Trajectories of the ensemble.
               
               N: int
               Number of ensemble members
               
           Returns:
           --------
               spread : ndarray-like(float)
               Ensemble spread as a function of time for all 3 spatial components.
               It is defined as the standard deviation of the ensemble members
               at everytime step.
               
               sol_ave: ndarray-like(float)
               Ensemble mean as a function of time for all 3 spatial components.
        
    
    """
    N = sol.shape[2]
    
    num_steps = sol.shape[0]
    
    spread = np.zeros((num_steps,3))
    
    S = np.zeros((N,3))    

    sol_ave = np.mean(sol, 2)
    
    for i in range(num_steps):
    
        for j in range(3):
    
            S[:,j] = np.array([sol[i,j,m] for m in range(N)])
    
            spread[i,j] = np.std(S[:,j])
            
    return spread, sol_ave


def prediction(
        rmse: np.ndarray,
        dt: float,
        threshold: float,
        ) -> Union[np.ndarray,float]:
    
    """ This function finds the value of the predictability time for each value 
        of the perturbation applied to the system. 
        
            Arguments:
            ----------
                rmse : ndarray-like(float)
                Root Mean Square Error as a function of time. It is defined as
                the square root of the sum of the squared differences between
                the corresponding components of the 2 trajectories.
                
                dt : float
                TIme step for the integration.
                
                threshold: float
                Value to be used as the threshold for the RMSE.
                
            Returns:
            --------
                pred_time : array-like(float)
                Measure of the chaotic nature of the system, i.e. the time 
                after which a perturbation to the IC of the system leads to
                a significative divergence of the solution with respect to 
                the unperturbed trajectory. 
                
                It is here defined as the time when the RMSE becomes greater
                than a certain threshold, chosen to be 0.5.
            
            Notes:
            ------
                It gives the predictability time both for the case where the 
                rmse is a 1D array that depends only on time and for the case 
                where it is a 2D array that depends also on the value of the 
                perturbation. It was done in order to allow the computation 
                for both an ensamble of perturbation and for a single trajectory.
                
    
    """
    if rmse.ndim == 1:
        
        if np.all(rmse < threshold):
        
            pred_time = 0.
            
            print('the RMSE is always smaller than 0.5 for the entire time ' +
            'window')
        
        else:
            
            pred_time = np.where(rmse > threshold)[0][0]*dt
        
            
        return pred_time
    
    else:
        
        num_of_perturbations = rmse.shape[1]
        pred_time1 = np.zeros(num_of_perturbations)
    
    
        for i in range(num_of_perturbations):
            
            if np.all(rmse[:,i] < threshold):
            
                pred_time1[i] = 0.
            
                print('for perturbation number', i, 
                      'the RMSE is always smaller than 0.5 for the entire time ' +
                      'window')
            
            else:
            
                pred_time1[i] = np.where(rmse[:,i] > threshold)[0][0]*dt
            
            
            
        return pred_time1
            
    
    
def func(
        x: np.ndarray,
        a: float,
        b: float,
        ) -> np.ndarray:
    
    """ This function returns a linear equation of the kind:
        y = a * x + b.
    
    Arguments:
    ----------
        x: np.ndarray(floats)
        Data.
        
        a: floats
        First parameter.
        
        b: floats
        Second parameter.
        
    Returns:
    --------
        y: np.ndarray(floats)
        Such that y = a * x + b.
    """
    y =  a * x + b
    return y
    

def fitting(func: Callable,
            x: np.ndarray,
            y: np.ndarray,
            best_guess_1: float,
            best_guess_2: float,
            )-> tuple:
    
    """This function produces a linear fit of the y data using a log scale for
    the x axis and gives a measure of the uncertainty. 
    
    It exploits scipy modules: scipy.optimize.curve_fit, 
    scipy.stats.multivariate_normal.rvs and scipy.stats.mstats.mquantiles.
    
    Arguments:
    ----------
        func:
        Function previously defined that returns a linear equation:
        y = a * x + b
        
        x: np.ndarray(floats)
        Data to be out in log scale.
        
        y: np.ndarray(floats)
        Data to be fitted.
        
        best_guess_1: float
        Best guess for the first parameter.
        
        best_guess_2: float
        Best guess for the second parameter.
        
        Returns:
        --------
            fit: np.ndarray(floats)
            The result of the fitting process.
            
            popt: np.ndarray(floats)
            Parameters resulting from the fitting process.
            
            p_low: np.ndarray(floats)
            An array containing the first calculated quantile.
            
            p_top:  np.ndarray(floats)
            An array containing the second calculated quantile.
            
        For further information about scipy modules, see the following links:
            
        ->https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        ->https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
        ->https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
        
            
        """
    
    popt, pcov = curve_fit(func, np.log10(x), y, p0=(best_guess_1, best_guess_2))
    
    fit = func(np.log10(x),*popt)
    
    p_seq = st.multivariate_normal.rvs(mean=popt, cov=pcov, size=50)
    
    fit_ = [func(np.log10(x),*params) for params in p_seq]
    
    p_low, p_top = mquantiles(fit_, prob=[0.025, 0.975], axis=0)
    
    return fit, popt, p_low, p_top
    











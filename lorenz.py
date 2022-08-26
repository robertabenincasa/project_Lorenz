# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:07:33 2022

@author: roberta benincasa
"""
import numpy as np

def read_parameters(par: str) -> np.ndarray:
    
    par = par.split(',')
    
    for i in range(len(par)):
        
        par[i] = float(par[i])
        
    return par


def lorenz(
           state_vector: np.ndarray, 
           t: np.ndarray,
           sigma: float,
           b: float,
           r1: float,
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
            
            t : array-like(float)
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
    y_dot = r1 * x - x * z - y 
    z_dot = x * y - b * z
    
    return [x_dot,y_dot,z_dot]


def perturbation(
                 init_cond: np.ndarray,
                 eps: np.ndarray,
                 ) -> np.ndarray:
    """ This function adds a perturbation to the first component of the 
    initial condition of the simulation.
    
        Arguments:
        ----------
            init_cond : array-like(float)
            Unperturbed initial values of the 3 variables of the system.
            
            eps : array-like(float)
            Three different values used to perturb the x-component of the 
            initial condition.
        
        Returns:
        --------
            IC : ndarray-like(float)
            Matrix where each rows represents a set of initial conditions. The 
            first row are the unperturbed ones.
            
        Notes:
        ------
            The number of perturbed ICs depends on the number of 
            perturbations. 
            The choice to perturb only the x-component is arbitrary.
    
    """
    
    IC = np.zeros((len(eps),3))
    IC[0,:] = init_cond
    for i in range(len(eps)):
        IC[i,:]=IC[0,:]+[eps[i],0.,0.]
    
    return IC

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
    
    delta_x = sol1[:,0] - sol2[:,0]
    
    return delta_x 



def RMSE(
        sol1: np.ndarray,
        sol2: np.ndarray,
        ) -> np.ndarray:
    
    """This function performs the calculation of the root mean square error 
       of the solution obtained from the perturbed ICs with respect to the 
       unperturbed one.
       
       It is a measure of the divergence of 2 trajectories of the same
       system given a slightly difference in the initial conditions adopted 
       for the integration.
       
       
           Arguments:
           ----------
               sol1 : ndarray-like(float)
               Unperturbed solution.
               
               sol2: ndarray-like(float)
               Perturbed solution.
               
           Returns:
           --------
               rmse : ndarray-like(float)
               Root Mean Square Error as a function of time. It is defined as
               the square root of the sum of the squared differences between
               the corresponding components of the 2 trajectories.
               
        
    
    """
    
    rsme = np.sqrt((sol1[:,0] - sol2[:,0])**2 + (sol1[:,1]-sol2[:,1])**2 + (sol1[:,2]-sol2[:,2])**2)
    
    return rsme


def prediction(
        error: np.ndarray,
        num_steps: int,
        dt: float,
        eps: np.ndarray,
        ) -> np.ndarray:
    
    """ This function finds the value of the prediction time for each value 
        of the perturbation applied to the system. 
        
            Arguments:
            ----------
                error : ndarray-like(float)
                Root Mean Square Error as a function of time. It is defined as
                the square root of the sum of the squared differences between
                the corresponding components of the 2 trajectories.
                
                num_steps : int
                Number of time steps for the integration.
                
                dt : float
                TIme step for the integration.
                
                eps : array-like(float)
                Vector of the possible perturbations that can be applied 
                to the x-component of the IC of the system.
                
            Returns:
            --------
                pred_time : array-like(float)
                Measure of the chaotic nature of the system, i.e. the time 
                after which a perturbation to the IC of the system leads to
                a significative divergence of the solution with respect to 
                the unperturbed trajectory. 
                
                It is here defined as the time when the RMSE becomes greater
                than a certain threshold, chosen to be 0.5 .
            
            Notes:
            ------
                The prediction time is here treated as an array since it 
                depends on the value of the perturbation.
                
    
    """
    pred_time = np.zeros(len(eps)-1)
    
    for i in range(1,len(eps)):
    
        for m in range(num_steps): 
        
            if error[m,i] > 0.5:
            
                pred_time[i-1] = m * dt 
            
                break 
        
        if np.all(error[:,i] < 0.5):
            
            print('for $/epsilon$ = ', eps[i], 
        'the RMSE is always smaller than 0.5 for the entire time window')
            
    return pred_time
            
    
    
    










# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:07:33 2022

@author: Lenovo
"""
import numpy as np

def lorenz(state_vector: float, t: float, sigma: float, b: float, r1: float, f: float) -> float: 
    """" This function returns the time derivative of the 3 variables x, y and 
    z as given by the Lorenz system, a simplified model of atmospheric 
    convection.
        
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
            [x_dot,y_dot,z_dot] : ndarray-like(float)
            Time derivative of the 3 variables x, y and z as obtained from the
            Lorenz system.
            
    """
    x,y,z=state_vector    
    
    x_dot = sigma * (y - x)
    y_dot = r1 * x - x * z - y + f
    z_dot = x * y - b * z
    
    return [x_dot,y_dot,z_dot]


def perturbation(init_cond: float, eps: float) -> float:
    """ This function adds a perturbation to the first component of the initial
    condition of the simulation.
    
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
            
    
    """
    
    IC = np.zeros((4,3))
    IC[0,:] = init_cond
    for i in range(len(eps)):
        IC[i,:]=IC[0,:]+[eps[i],0.,0.]
    
    return IC

def difference(sol1: float, sol2: float) -> float:
    
    delta_x = sol1[:,0] - sol2[:,0]
    
    return delta_x

def RSME(sol1: float,sol2: float) -> float:
    
    rsme = np.sqrt((sol1[:,0] - sol2[:,0])**2 + (sol1[:,1]-sol2[:,1])**2 + (sol1[:,2]-sol2[:,2])**2)
    
    return rsme














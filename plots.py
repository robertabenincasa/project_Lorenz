# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:40:54 2022

@author: Lenovo
"""

import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
import numpy as np

def xzgraph(
        sol: np.ndarray,
        r : float):
    """ This function produces a plot of the solution of the integration of the 
    Lorenz system in the plane x, z. 
        
        Arguments:
        ----------
        
            sol : ndarray-like(float) 
            It is the trajectory of the system in the 3 directions.
            
            r: scalar(float)
            The value of the parameter r of the Lorenz system used in the 
            simulation, i.e. the Rayleigh number. 
            
        Notes:
        ------
        The argument r is treated here as a variable in order to obtain the 
        graph of the trajectory for each value of r used in the simulation.
        
        Raises:
        -------
        ValueError if the 2 components of the solutions do not have the same 
        first dimension.
            
    """
    rho = r
    fig,(ax)=plt.subplots(1,1,figsize=(8,6))
    ax.plot(sol[:,0], sol[:,2],'k', marker='.',markersize=1, label='L(IC1,t)')
    ax.set_ylim([0,50])
    ax.legend(loc='best')
    ax.set_title('Solution of the numerical integration - r = %i'%rho)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.grid()
    plt.show()
    
def plot_3dsolution(
        sol: np.ndarray,
        r: np.ndarray):
    
    rho = r
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(sol[:,0], sol[:,1],sol[:,2], 'black', marker='.',markersize=0.2)
    ax.set_title('Solution of the numerical integration - r = %i' %rho)






def plot_difference(
        diff: np.ndarray,
        t: np.ndarray,
        r: float):
    
    rho = r
    fig,(ax)=plt.subplots(1,1,figsize=(8,6))
    ax.plot(t, diff,'k', marker='.',markersize=1, label='')
    #ax.set_ylim([0,50])
    ax.legend(loc='best')
    ax.set_title('Difference - r = %i'%rho)
    ax.set_xlabel('t')
    ax.grid()
    plt.show()

def plot_rmse(
        rsme: np.ndarray,
        t: np.ndarray,
        r: float,
        e: float,
        pred_time: float):
    
    rho = r
    fig,(ax)=plt.subplots(1,1,figsize=(8,6))
    ax.plot(t, rsme,'k', marker='.',markersize=1, label='eps')
    ax.axvline(pred_time, color = 'red', label = 'prediction time')
    ax.legend(loc='best')
    ax.set_title('Root Mean Square Error - r = %i'%rho)
    ax.set_xlabel('t')
    ax.grid()
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:40:54 2022

@author: roberta benincasa
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
    
    fig,(ax)=plt.subplots(1,1,figsize=(10,8))
    ax.grid()
    
    ax.plot(sol[:,0], sol[:,2],'k', marker='.',markersize=1, label='L(IC0,t)')
    ax.set_title('Solution of the numerical integration - r = %i'%r)
    
    ax.set_ylim([0,50])
    ax.legend(loc='best')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    
    plt.show()
    
def plot_3dsolution(
        sol: np.ndarray,
        r: np.ndarray):
    
    fig = plt.figure(figsize = (10,8))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.plot3D(sol[:,0], sol[:,1],sol[:,2], 'black', marker='.',markersize=0.5)
    ax.set_title('Solution of the numerical integration - r = %i' %r)
   
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    plt.show()



def plot_difference(
        diff: np.ndarray,
        t: np.ndarray,
        r: float):
    
    
    fig,(ax)=plt.subplots(1,1,figsize=(10,8))
    ax.grid()
    
    ax.plot(t, diff,'k', marker='.',markersize=1, label = 'L(IC0,t) - L(IC1,t)')
    ax.set_title('Difference - r = %i'%r)
    
    ax.set_xlabel('t')
    ax.legend(loc='best')
    
    plt.show()

def plot_rmse(
        rmse: np.ndarray,
        t: np.ndarray,
        r: float,
        e: float,
        pred_time: float):
    
    
    fig,(ax,bx)=plt.subplots(2,1,figsize=(10,8))
    plt.subplots_adjust(wspace=2, hspace=0.5)
    ax.grid()
    bx.grid()
    
        
    ax.plot(t, rmse,'k', marker='.',markersize=1, 
            label='eps = '+ np.format_float_scientific(e))
    ax.axvline(pred_time, color = 'red', 
               label = 'prediction time = '+ np.format_float_scientific(pred_time))
    ax.set_title('Root Mean Square Error - r = %i'%r)
    
    bx.semilogy(t, rmse,'k',marker='.',markersize=1, 
                label='eps = '+ np.format_float_scientific(e))
    bx.axvline(pred_time, color = 'red', 
               label = 'prediction time = '+ np.format_float_scientific(pred_time))
    bx.set_title('Root Mean Square Error - Log scale - r = %i'%r)
    
    ax.legend(loc='best')
    ax.set_xlabel('t')
    
    bx.legend(loc='best')
    bx.set_xlabel('t')
    
    plt.show()
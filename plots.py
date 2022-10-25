# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:40:54 2022

@author: roberta benincasa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def xzgraph(
        sol: np.ndarray,
        r : float):
    
    """ This function produces a plot of the solution of the integration of the 
    Lorenz system in (x,z) plane.
    
        Arguments:
        ----------
            sol: np.ndarray(floats)
            Solution of the integration of the Lorenz system. The first 
            dimension represent time while the second is used to specify the 
            variable (x,y or z).
            
            r: float
            Parameter r of the Lorenz system used in the integration.
       
    """
    
    fig,(ax)=plt.subplots(1,1,figsize=(8,6))
    ax.grid()
    
    ax.plot(sol[:,0], sol[:,2],'indigo', marker='.',markersize=1, label='L(IC0,t)')
    ax.set_title('Solution of the numerical integration - r = %i'%r)
    
    ax.set_ylim([0,50])
    ax.legend(loc='best')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    
    
def plot_3dsolution(
        sol: np.ndarray,
        r: float):
    
    """ This function produces a 3D plot of the solution of the integration of
    the Lorenz system.
    
        Arguments:
        ----------
            sol: np.ndarray(floats)
            Solution of the integration of the Lorenz system. The first 
            dimension represent time while the second is used to specify the 
            variable (x,y or z).
            
            r: float
            Parameter r of the Lorenz system used in the integration.
        
        """
    
    plt.figure(figsize = (8,6))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.plot3D(sol[:,0], sol[:,1],sol[:,2], 'indigo', marker='.',markersize=0.5)
    ax.set_title('Solution of the numerical integration - r = %i' %r,size = 20)
   
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    

    
    
def plot_animation(sol: np.ndarray,
                   sol1: np.ndarray,
                   r: float,
                   eps: float,
                   ) -> animation.FuncAnimation:
    
    """ This function produces an animation of the solution of the integration 
    of the Lorenz system for both the perturbed and unperturbed one, both as 
    functions of time.
    
        Arguments:
        ----------
            sol: np.ndarray(floats)
            Unperturbed solution of the integration of the Lorenz system. The first 
            dimension represent time while the second is used to specify the 
            variable (x,y or z).
            
            sol1: np.ndarray(floats)
            Perturbed solution of the integration of the Lorenz system. The first 
            dimension represent time while the second is used to specify the 
            variable (x,y or z).
            
            r: float
            Parameter r of the Lorenz system used in the integration.
            
            eps: float
            Value of the perturbation applied to the intial condition.
            
        Returns:
        --------
            anim: matplotlib.animation.FuncAnimation
            Animation.
   """
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot( projection='3d')
    
    lines = []
    colors = ['purple','royalblue']
    label = ['no perturbations','$\epsilon$ =' + np.format_float_scientific(eps)]
    
    for index in range(2):
        lobj, = ax.plot( [], [], [], color=colors[index],lw=1.5,
                        label = label[index])
        lines.append(lobj)
    
    
    def init(): 
    
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
            return 
    
    def animate(num): 
        num = (40 * num) % sol.shape[0]
        xlist = [sol[:num,0],sol1[:num,0]]
        ylist = [sol[:num,1],sol1[:num,1]]
        zlist = [sol[:num,2],sol1[:num,2]]
        
        
        for lnum,line in enumerate(lines):
          
            line.set_data(xlist[lnum],ylist[lnum])    
            line.set_3d_properties(zlist[lnum])    
    
        return 
    
    
    ax.set_title('Solution of the numerical integration -'+'\n'+' r = %i'%r
               ,size=20)

    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_zlim(5,50)
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    ax.legend(loc='best')
   
# Creating the Animation object

    anim = animation.FuncAnimation(fig, animate, init_func=init(), 
                            frames=300, interval=2, blit=False)
    
    return anim


    
def plot_difference(
        diff: np.ndarray,
        diff1: np.ndarray,
        t: np.ndarray,
        eps: float,
        ):
    
    """ This function produces a plot of the difference between the unperturbed 
    and the perturbed solution of the integration of the Lorenz system along 
    a single component for both value of r, as a function of time.
    
        Argumets:
        ---------
            diff: np.ndarray(floats)
            Difference between the unperturbed and the perturbed solution of 
            the integration of the Lorenz system along a single component for 
            r = 28.
            
            diff1: np.ndarray(floats)
            Difference between the unperturbed and the perturbed solution of 
            the integration of the Lorenz system along a single component for 
            r = 9.
            
            t: np.ndarray(floats)
            Time.
            
            eps: float
            Value of the perturbation applied to the intial condition.
                    
   """
    
    fig,((ax),(ax1))=plt.subplots(2,1, sharex=True, figsize=(8,6))
    
    ax.grid()
    ax.plot(t, diff,'cornflowerblue', marker='.',markersize=1, label = 'r = 28')
    ax.set_title('Difference between x-components - $\epsilon$ = '+ np.format_float_scientific(eps))
    ax.legend(loc='best')
    
    ax1.grid()
    ax1.plot(t, diff1,'purple', marker='.',markersize=1, label = 'r = 9')
    ax1.legend(loc='best')
    ax1.set_xlabel('t')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,-10))
    
    
    
    
def plot_rmse(
        rmse: np.ndarray,
        t: np.ndarray,
        r: float,
        e: float,
        pred_time: float):
    
    """ This function produces a plot of the RMSE as a function of time 
    both in log scale and in linear scale. The value of the predictability time
    is highlighted with a vertical line too.
    
        Arguments:
        ----------
            rmse: np.ndarray(floats)
            RMSE as a function of time.
            
            t: np.ndarray(floats)
            Time.
            
            r: float
            Parameter r of the Lorenz system used in the integration.
            
            e: float
            Value of the perturbation applied to the intial condition.
            
            pred_time: float
            Value of the predictability time for the chosen values of the 
            perturbation and of r.
            
        """
    
    
    fig,(ax,bx)=plt.subplots(2,1, sharex=True, figsize=(8,6))
    plt.subplots_adjust(wspace=2, hspace=0.5)
    ax.grid()
    bx.grid()
    
        
    ax.plot(t, rmse,'cornflowerblue', marker='.',markersize=1, 
            label='$\epsilon$ = '+ np.format_float_scientific(e))
    ax.axvline(pred_time, color = 'purple', 
               label = 'prediction time = '+ np.format_float_scientific(pred_time))
    ax.set_title('Root Mean Square Error - r = %i'%r)
    
    bx.semilogy(t, rmse,'cornflowerblue',marker='.',markersize=1, 
                label='$\epsilon$ = '+ np.format_float_scientific(e))
    bx.axvline(pred_time, color = 'purple', 
               label = 'prediction time = '+ np.format_float_scientific(pred_time))
    bx.set_title('Root Mean Square Error - Log scale - r = %i'%r)
    
    bx.legend(loc='best')
    bx.set_xlabel('t')


    
def plot_ensemble(
        L: np.ndarray,
        R: np.ndarray,
        t: np.ndarray):
    
    """ This function produces a plot of the RMSE of the ensemble mean (L) and
     of the mean of the RMSEs of the ensemble, both as functions of time.
     
         Arguments:
         ----------
             L: np.ndarray(floats)
             RMSE of the ensemble mean as a function of time.
             
             R: np.ndarray(floats)
             The mean RMSE of the ensemble as a function of time.
             
             t: np.ndarray(floats)
             Time.
             
             
  """
    fig,(ax)=plt.subplots(1,1,figsize=(10,4))
    
    ax.grid()
    ax.plot(t, L,'royalblue', marker='.',markersize=1, label = 'L')
    ax.plot(t, R,'skyblue', marker='.',markersize=1, label = 'R')
    ax.set_title('RMSE of the Ensemble mean vs mean RMSE ')
    ax.set_xlabel('t')
    ax.legend(loc='best')
    
    
    
def plot_ensemble_trajectories(
        sol: np.ndarray,
        S: np.ndarray,
        t: np.ndarray):
    
    """ This function produces a plot of the ensemble mean as a function of 
    time for each of the 3 variable: x, y and z.
    The ensemble spread is indicated as a shaded area. 
    
        Arguments:
        ----------
            sol: np.ndarray(floats)
            Ensemble mean of the solutions of the integration of the Lorenz 
            system. The first dimension represent time while the second is used 
            to specify the variable (x,y or z).
            
            S: np.ndarray(floats)
            Ensemble spread of the solutions of the integration of the Lorenz 
            system. The first dimension represent time while the second is used 
            to specify the variable (x,y or z).
            
            t: np.ndarray(floats)
            Time.
            
            
   """
    
    fig,((ax),(ax1),(ax2))=plt.subplots(3,1, sharex=True, figsize=(10,8))
    
    ax.grid()
    ax.plot(t, sol[:,0] ,'royalblue', marker='.',markersize=1, label = 'X')
    ax.fill_between(t,sol[:,0] - S[:,0],sol[:,0] + S[:,0],alpha=0.3, facecolor='royalblue')
    ax.set_title('Ensemble mean and ensemble spread', size = 20)
    ax.legend(loc='best')
    
    ax1.grid()
    ax1.plot(t,sol[:,1] ,'purple', marker='.',markersize=1,label = 'Y')
    ax1.fill_between(t,sol[:,1] - S[:,1],sol[:,1] + S[:,1],alpha=0.3, facecolor='purple')
    ax1.legend(loc='best')
    
    
    ax2.grid()
    ax2.plot(t,sol[:,2] ,'violet', marker='.',markersize=1, label = 'Z')
    ax2.fill_between(t,sol[:,2] - S[:,2],sol[:,2] + S[:,2],alpha=0.3, facecolor='violet')
    ax2.set_xlabel('t')
    ax2.legend(loc='best')
    
    
    
    
def pred_time_vs_perturbation(
        pred_time: np.ndarray,
        eps: np.ndarray,
        fit: np.ndarray,
        popt: np.ndarray,
        p_low: np.ndarray,
        p_top: np.ndarray,
        fit1: np.ndarray,
        popt1: np.ndarray,
        p_low1: np.ndarray,
        p_top1: np.ndarray,
        ):

    """ This function produces a plot of the 2 fit performed on the relation
    between the predictability time and the perturbation applied. The data are
    reported as points, whereas the uncertainty as a shaded area.
    
        Arguments:
        ----------
            pred_time: np.ndarray(floats)
            Predictability time for each value of the perturbation.
            
            eps: np.ndarray(floats)
            Array with the values of the pertubation.
            
            fit, fit1: np.ndarray(floats)
            First and second fit, i.e. y =  ax + b.
            
            popt, popt1: np.ndarray(floats)
            Array with the values of the parameters a and b of the first and 
            second fit, respectively.
            
            p_low, p_low1: np.ndarray(floats)
            Lower limit for the uncertainty for the first and 
            second fit, respectively.
            
            p_top, p_top1: np.ndarray(floats)
            Upper limit for the uncertainty for the first and 
            second fit, respectively.
            
        Note:
        -----
            For further information, please see the following link:
                
            ->https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
            
            
            
            
            
"""
    fig,(ax)=plt.subplots(1,1,figsize=(8,6))
    ax.grid()
    
    plt.scatter(eps,pred_time,c='indigo', label = 'data')
    plt.plot(eps, fit, 'purple',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.plot(eps[0:4], fit1, 'cornflowerblue',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt1))
    plt.fill_between(eps, p_low, p_top, alpha=0.1, color='violet')
    plt.fill_between(eps[0:4], p_low1, p_top1, alpha=0.1, color='royalblue')
    ax.set_xscale('log')
    ax.set_title('Predictability time', size = 20)
    ax.legend(loc='best')
    
    ax.set_xlabel('Perturbation')

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:40:54 2022

@author: roberta benincasa
"""
import configparser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




config = configparser.ConfigParser()
config.read('config.ini')

path = config.get('Paths to files', 'path')

def xzgraph(
        sol: np.ndarray,
        r : float):

    
    fig,(ax)=plt.subplots(1,1,figsize=(8,6))
    ax.grid()
    
    ax.plot(sol[:,0], sol[:,2],'indigo', marker='.',markersize=1, label='L(IC0,t)')
    ax.set_title('Solution of the numerical integration - r = %i'%r)
    
    ax.set_ylim([0,50])
    ax.legend(loc='best')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    
    plt.savefig(path +'/xzplane_plot_r=%i'%r + '.png')
    plt.show()
    
def plot_3dsolution(
        sol: np.ndarray,
        r: float):
    
    plt.figure(figsize = (8,6))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.plot3D(sol[:,0], sol[:,1],sol[:,2], 'indigo', marker='.',markersize=0.5)
    ax.set_title('Solution of the numerical integration - r = %i' %r,size = 20)
   
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    
    plt.savefig(path + '/3Dplot_r=%i'%r + '.png')
    plt.show()
    
    
def plot_animation(sol, sol1, r, eps):
    
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
    
    anim.save(path + '/animation.gif')
   # plt.show()
    plt.savefig(path + '/3Dplot_r=%i'%r + '_eps=' + np.format_float_scientific(eps)+'.png')
    
def plot_difference(
        diff: np.ndarray,
        diff1: np.ndarray,
        t: np.ndarray,
        eps: float,
        ):
    
    
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
                      
    
    
    plt.savefig(path + '/difference.png')
    plt.show()
    
def plot_rmse(
        rmse: np.ndarray,
        t: np.ndarray,
        r: float,
        e: float,
        pred_time: float):
    
    
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
    
    plt.savefig(path + '/rmse_r=%i'%r + '_eps='+ np.format_float_scientific(e)+'.png')
    plt.show()
    
def plot_ensemble(
        L: np.ndarray,
        R: np.ndarray,
        t: np.ndarray):
    
    fig,(ax)=plt.subplots(1,1,figsize=(10,4))
    
    ax.grid()
    ax.plot(t, L,'royalblue', marker='.',markersize=1, label = 'L')
    ax.plot(t, R,'skyblue', marker='.',markersize=1, label = 'R')
    ax.set_title('RMSE of the Ensemble mean vs mean RMSE ')
    ax.set_xlabel('t')
    ax.legend(loc='best')
    
    plt.savefig(path + '/ensemble.png')
    plt.show()
    
    
def plot_ensemble_trajectories(
        sol: np.ndarray,
        S: np.ndarray,
        t: np.ndarray):
    
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
    
    
    plt.savefig(path + '/ensemble_trajectories.png')
    plt.show()
    
    
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
    
    plt.savefig(path +'/pred_time_vs_perturbation.png')
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:31:06 2022

@author: Lenovo
"""
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
from tabulate import tabulate
import plots
import lorenz


default_file = 'config.ini' #default configuration file


config = configparser.ConfigParser()
configuration_file = lorenz.reading_configuration_file(default_file)
config.read(configuration_file)


#--------------------------READING CONFIGURATION FILE-------------------------#


path_plots = config.get('Paths to files', 'path_plots')

path_data = config.get('Paths to files', 'path_data')

r1 = float(config.get('Parameters', 'r1')) #chaotic solution

r2 = float(config.get('Parameters', 'r2')) #non-chaotic solution

eps1 = config.get('Perturbations', 'eps') #perturbations
eps = np.array(eps1.split(','), dtype=float)

num_steps = int(config.get('Integration settings', 'num_steps'))

dt = float(config.get('Integration settings', 'dt'))

idx_anim = int(config.get('Plotting','which_eps_for_animation'))

idx_diff = int(config.get('Plotting','which_eps_for_difference'))

t = np.linspace(0,num_steps,num_steps)*dt #time variable


#--------------------------------READING DATA---------------------------------#

sol_chaotic = np.load(path_data+'/sol_chaotic.npy')
sol_non_chaotic = np.load(path_data+'/sol_non_chaotic.npy')

delta_chaotic = np.load(path_data+'/delta_chaotic.npy')
delta_non_chaotic = np.load(path_data+'/delta_non_chaotic.npy')
rmse = np.load(path_data+'/rmse.npy')
pred_time = np.load(path_data+'/pred_time.npy')

fit = np.load(path_data+'/fit.npy')
popt = np.load(path_data+'/popt.npy')
p_low = np.load(path_data+'/p_low', allow_pickle= True)
p_top = np.load(path_data+'/p_top', allow_pickle= True)

fit1 = np.load(path_data+'/fit1.npy')
popt1 = np.load(path_data+'/popt1.npy')
p_low1 = np.load(path_data+'/p_low1', allow_pickle= True)
p_top1 = np.load(path_data+'/p_top1', allow_pickle= True)

sol_ens = np.load(path_data+'/ensemble_solution.npy')
spread = np.load(path_data+'/ensemble_spread.npy')
sol_ave = np.load(path_data+'/ensemble_mean.npy')
L = np.load(path_data+'/L.npy')
R = np.load(path_data+'/R.npy')
pred_time_R = np.load(path_data+'/pred_time_R.npy')
pred_time_L = np.load(path_data+'/pred_time_L.npy')


#----------------------------------PLOTTING-----------------------------------#

#PLOTTING both chaotic and non-chaotic solution for
#the unpertubed case in the x,z plane

plots.xzgraph(sol_chaotic[:,:,0],r1) 
plt.savefig(path_plots +'/xzplane_plot_r=%i'%r1 + '.png')
plt.show()

plots.xzgraph(sol_non_chaotic[:,:,0],r2) 
plt.savefig(path_plots +'/xzplane_plot_r=%i'%r2 + '.png')
plt.show()

plots.plot_3dsolution(sol_non_chaotic[:,:,0],r2)
plt.savefig(path_plots + '/3Dplot_r=%i'%r2 + '.png')
plt.show()

plots.plot_3dsolution(sol_chaotic[:,:,0],r1)
plt.savefig(path_plots + '/3Dplot_r=%i'%r1 + '.png')
plt.show()

#3D animation of the chaotic solution for the unperturbed and a 
#perturbed case 
print('\n')
print('-----------------------Preparing the animation------------------------')
print('---------------This operation may require a few seconds---------------')

anim = plots.plot_animation(sol_chaotic[:,:,0],sol_chaotic[:,:,9],r1,
                              eps[idx_anim])

anim.save(path_plots + '/animation.gif')
plt.show()

#PLOTTING the results of the analysis:

plots.plot_difference(delta_chaotic,delta_non_chaotic,t,eps[idx_diff]) 
plt.savefig(path_plots + '/difference.png')
plt.show()

for i in range(0,len(eps),2): 
   
    plots.plot_rmse(rmse[:,i],t, r1, eps[i], pred_time[i])
    plt.savefig(path_plots + '/rmse_r=%i'%r1 + '_eps='+
              np.format_float_scientific(eps[i])+'.png')
    plt.show()
 
#FITTING: predictability time vs applied perturbation.
#The aim is to verify their logarithmic dependence.
#There are 2 separate fit: one for all the available values of the perturbation 
#and the other just for the infinitesimal values

plots.pred_time_vs_perturbation(pred_time, eps, fit, popt, p_low, p_top, 
                                    fit1, popt1, p_low1, p_top1)
plt.savefig(path_plots +'/pred_time_vs_perturbation.png')
plt.show()

#PLOTTING the results of the ensemble analysis 
plots.plot_ensemble(L,R,t)
plt.savefig(path_plots + '/ensemble.png')
plt.show()

plots.plot_ensemble_trajectories(sol_ave,spread,t)
plt.savefig(path_plots + '/ensemble_trajectories.png')
plt.show()



#-----------------------------------TABLES------------------------------------#


#Creating a table with the values of the perturbation and
#of the corresponding prediction times 


print('\n')
print('Lorenz system with r = 28:')
print('\n')

data = np.column_stack((eps, pred_time))
col_names = ["Perturbation", "Predictability time"]
print('Single forecast:')
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

data1 = np.column_stack((pred_time_L, pred_time_R))
col_names1 = ["Prediction time L", "Predictability time R"]
print('Ensemble forecast:')
print(tabulate(data1, headers=col_names1, tablefmt="fancy_grid"))

data2 = np.column_stack((popt[0], popt[1]))
col_names = ["a", "b"]
print('Fit:')
print(tabulate(data2, headers=col_names, tablefmt="fancy_grid"))

data3 = np.column_stack((popt1[0], popt1[1]))
col_names = ["a", "b"]
print('Fit1:')
print(tabulate(data3, headers=col_names, tablefmt="fancy_grid"))

#-----------------------------Saving to png files:----------------------------#

df = pd.DataFrame(data = {'Perturbation': eps,'Predictability time': pred_time})
dfi.export(df, path_plots + '/table_predtime.png',fontsize = 30)


df1 = pd.DataFrame(data = {'Predictability time L': pred_time_L, 
                           'Predictability time R': pred_time_R},index= [1])
dfi.export(df1, path_plots + '/table_LR.png',fontsize = 30)

print('\n')
print('Plots and tables are now available in the folder: output')
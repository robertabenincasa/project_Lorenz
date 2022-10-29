# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:58:03 2022

@author: roberta benincasa
"""
import configparser
import numpy as np
import lorenz


#--------------------------READING CONFIGURATION FILE-------------------------#



default_file = 'config.ini' #default configuration file
configuration_file = lorenz.reading_configuration_file(default_file)

config = configparser.ConfigParser()
config.read(configuration_file)

path_data = config.get('Paths to files', 'path_data')

#---------------------------Lorenz System Parameters--------------------------#

sigma = float(config.get('Parameters', 'sigma'))

b = float(config.get('Parameters', 'b'))

r1 = float(config.get('Parameters', 'r1')) #chaotic solution

r2 = float(config.get('Parameters', 'r2')) #non-chaotic solution


set_A, set_B = [sigma, b, r1], [sigma, b, r2]


#--------------------------Integration Parameters-----------------------------#

num_steps = int(config.get('Integration settings', 'num_steps'))

dt = float(config.get('Integration settings', 'dt'))

N = int(config.get('Integration settings', 'N'))

IC01 = config.get('Initial condition', 'IC') #initial condition
IC0 = np.array(IC01.split(','), dtype=float)


#----------------------------Analysis Parameters------------------------------#

which_variable = int(config.get('Perturbations', 'which_variable'))

eps1 = config.get('Perturbations', 'eps') #perturbations
eps = np.array(eps1.split(','), dtype=float)
    
random_seed = int(config.get('Integration settings', 'Random seed'))

alpha = float(config.get('Analysis', 'Threshold'))

idx_diff = int(config.get('Plotting','which_eps_for_difference'))


#------------------------------INTEGRATION------------------------------------#


t = np.linspace(0,num_steps,num_steps)*dt #time variable

IC = lorenz.perturbation(IC0,eps,which_variable) #perturbed initial conditions


#The following are the solution for each time step,
#for each variable, for each IC and for each value of r

sol_chaotic = lorenz.integration_Lorenz_system(lorenz.lorenz, num_steps, t, IC, 
                                               set_A)

sol_non_chaotic = lorenz.integration_Lorenz_system(lorenz.lorenz, num_steps, t,
                                                   IC, set_B)

sol_true = sol_chaotic[:,:,0]

#---------------------------------ANALYSIS------------------------------------#


#The difference is performed only between the solution of the unperturbed 
#case and the one of the first perturbed case, as a preliminary analysis.
#The difference is calculated for both chaotic and non-chaotic solution.

delta_chaotic = lorenz.difference(sol_true[:,which_variable], 
                                  sol_chaotic[:,which_variable,idx_diff]) 

delta_non_chaotic = lorenz.difference(sol_non_chaotic[:,which_variable,0], 
                                    sol_non_chaotic[:,which_variable,idx_diff])

#The RMSE and the prediction time are calculated for each perturbed case
#with r = 28 because it would be trivial for r = 9.

rmse = lorenz.RMSE(sol_chaotic)

pred_time = lorenz.prediction(rmse, dt, alpha)

fit, popt, p_low, p_top = lorenz.fitting(lorenz.func,eps,pred_time, -1.1, 10)[0:4]
fit1, popt1, p_low1, p_top1 = lorenz.fitting(lorenz.func,eps[0:4],
                                             pred_time[0:4], -1.1, 10)[0:4]


#------------------------------ENSEMBLE INTEGRATION---------------------------#


#Same procedure but with an ensemble of perturbations: showing how to improve
#the prediction!

eps_ens = lorenz.generate_random_perturbation(random_seed, N)

IC_ens = lorenz.perturbation(IC0, eps_ens, which_variable)

sol_ens =  lorenz.integration_Lorenz_system(lorenz.lorenz, num_steps, t, 
                                            IC_ens, set_A)


#------------------------------ENSEMBLE ANALYSIS------------------------------#


rmse_ens =  lorenz.RMSE(sol_ens)

spread, sol_ave = lorenz.ensemble(sol_ens) 
 
#R is the mean of the RMSEs and L is the RMSE of the mean.
#The aim is to compare the 2 and show how introducing an ensemble of simulations
#allows to halve the RMSE with respect to the one relative to a single simulation.

R, L = lorenz.calculating_L_and_R(sol_true, sol_ave, rmse_ens)

pred_time_L = lorenz.prediction(L, dt, alpha)
pred_time_R = lorenz.prediction(R, dt, alpha)
    

#-------------------------------SAVING TO FILE--------------------------------#


np.save(path_data+'/sol_chaotic',sol_chaotic)
np.save(path_data+'/sol_non_chaotic',sol_non_chaotic)

np.save(path_data+'/delta_chaotic',delta_chaotic)
np.save(path_data+'/delta_non_chaotic',delta_non_chaotic)
np.save(path_data+'/rmse',rmse)
np.save(path_data+'/pred_time',pred_time)

np.save(path_data+'/fit',fit) 
np.save(path_data+'/popt',popt)
p_low.dump(path_data+'/p_low')
p_top.dump(path_data+'/p_top')

np.save(path_data+'/fit1',fit1) 
np.save(path_data+'/popt1',popt1)
p_low1.dump(path_data+'/p_low1')
p_top1.dump(path_data+'/p_top1')

np.save(path_data+'/ensemble_solution', sol_ens)
np.save(path_data+'/ensemble_spread', spread)
np.save(path_data+'/ensemble_mean', sol_ave)
np.save(path_data+'/L',L)
np.save(path_data+'/R',R)
np.save(path_data+'/pred_time_R',pred_time_R)
np.save(path_data+'/pred_time_L',pred_time_L)



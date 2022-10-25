# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 17:11:48 2022

@author: Lenovo
"""

from configparser import ConfigParser
config = ConfigParser()

config.read('config.ini')

config.add_section('Integration settings')

config.set('Integration settings', 'num_steps', '12000')
config.set('Integration settings', 'dt', '0.005')
config.set('Integration settings', 'N', '100')
config.set('Integration settings','Random seed','42')

config.add_section('Parameters')

config.set('Parameters', 'sigma', '10.0')
config.set('Parameters', 'b', str(8./3.))
config.set('Parameters', 'r1', '28.0')
config.set('Parameters', 'r2', '9.0')

config.add_section('Initial condition')

config.set('Initial condition', 'IC', '9., 10., 20.')

config.add_section('Perturbations')

config.set('Perturbations','which_variable','0')
config.set('Perturbations', 'eps', '1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5 ,'+
           '1E-4, 1E-3, 1E-2, 1E-1, 1.0')

config.add_section('Analysis')

config.set('Analysis', 'Threshold', '0.5')

config.add_section('Paths to files')

config.set('Paths to files', 'path_data', './output/data')
config.set('Paths to files', 'path_plots', './output/plots')

config.add_section('Plotting')

config.set('Plotting', 'which_eps_for_difference', '7')
config.set('Plotting', 'which_eps_for_animation', '9')

with open('config.ini', 'w') as f:
    config.write(f)
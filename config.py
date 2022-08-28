# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 17:11:48 2022

@author: Lenovo
"""

from configparser import ConfigParser
import numpy as np
config = ConfigParser()

#eps = np.array([ 0. , 1E-5 , 1E-3 , 1.0 ])
#IC0 = np.array([ 9. , 10. , 20. ])

config.read('config.ini')

config.add_section('Integration settings')

config.set('Integration settings', 'num_steps', '12000')
config.set('Integration settings', 'dt', '0.005')

config.add_section('Parameters')

config.set('Parameters', 'sigma', '10.0')
config.set('Parameters', 'b', str(8./3.))
config.set('Parameters', 'r1', '28.0')
config.set('Parameters', 'r2', '9.0')
config.add_section('Initial condition')

config.set('Initial condition', 'IC', '9. , 10. , 20.')

config.add_section('Perturbations')

config.set('Perturbations', 'eps', '0. , 1E-5 , 1E-3 , 1.0')

config.add_section('Paths to files')

config.set('Paths to files', 'path', '/Users/Lenovo/Desktop/SCproject/output')

with open('config.ini', 'w') as f:
    config.write(f)
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
#config.add_section('Parameters')
#config.set('Parameters', 'sigma', '10.0')
#config.set('Parameters', 'b', '8./3.')
#config.set('Parameters', 'r1', '28.0')
#config.set('Parameters', 'r2', '9.0')
#config.add_section('Initial condition')
#config.set('Initial condition', 'IC', '9.,10.,20.')
#config.add_section('Perturbations')
#config.set('Perturbations', 'eps',  '1E-10,1E-5,1.0')

with open('config.ini', 'w') as f:
    config.write(f)
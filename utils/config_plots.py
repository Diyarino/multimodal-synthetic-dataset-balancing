# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:02:08 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

#%% imports
import matplotlib.pyplot as plt
from distutils.spawn import find_executable

#%% configurations

def configure_plt(check_latex = True):
        """
        Set Font sizes for plots.
    
        Parameters
        ----------
        check_latex : bool, optional
            Use LaTex-mode (if available). The default is True.
    
        Returns
        -------
        None.
    
        """
        
        if check_latex:
            
            if find_executable('latex'):
                plt.rc('text', usetex=True)
            else:
                plt.rc('text', usetex=False)
        plt.rc('font',family='Times New Roman')
        plt.rcParams.update({'figure.max_open_warning': 0})
        
        small_size = 13
        small_medium = 14
        medium_size = 16
        big_medium = 18
        big_size = 20
        
        plt.rc('font', size = small_size)          # controls default text sizes
        plt.rc('axes', titlesize = big_medium)     # fontsize of the axes title
        plt.rc('axes', labelsize = medium_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize = small_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize = small_size)    # fontsize of the tick labels
        plt.rc('legend', fontsize = small_medium)    # legend fontsize
        plt.rc('figure', titlesize = big_size)  # fontsize of the figure title
        
        plt.rc('grid', c='0.5', ls='-', lw=0.5)
        plt.grid(True)
        plt.tight_layout()
        plt.close()
        
#%% test

if "__name__" == "__main__":
    configure_plt()
    
        
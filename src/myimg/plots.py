'''
Module: myimg.plots
-------------------

Auxiliary module with functions for plotting.

>>> # Simple example how to employ myimg.plots
>>> # (the module is imported and used within myimg.api
>>> import myimg.api as mi
>>> mi.Plots.set_plot_parameters(size=(8,8), dpi=100)
'''

import matplotlib.pyplot as plt

class Plots:
    '''
    This (non-OO) class collects several useful functions for plotting.
    
    The 1st group of functions = setting/restoring global plot parameters:
    
    * set_plot_params, save_plot_params = setting/saving the plot params
    * restore_previous_params, restore_saved_params = restoring plot params
    
    The 2nd group of functions = specific plots for ELD/XRD processing:
        
    * plot_final_eld_and_xrd
      = usual final plot in package EDIFF
    * plot_radial_distributions
      = optional final plot for sister package STEMDIFF
      
    Moreover the class employs two (private) class variables:
        
    * _previous_rcParams = employed in restore_previous_params
    * _saved_rcParams    = employed in restore_saved_params
    '''

    _previous_rcParams = None
    _saved_rcParams = None

    @staticmethod    
    def set_plot_params(size=(8,6), dpi=100, fontsize=8, 
            my_defaults=True, my_rcParams=None):
        '''
        Set global plot parameters (mostly for plotting in Jupyter).
    
        Parameters
        ----------
        size : tuple of two floats, optional, the default is (8,6)
            Size of the figure (width, height) in [cm].
        dpi : int, optional, the defalut is 100
            DPI of the figure.
        fontsize : int, optional, the default is 8
            Size of the font used in figure labels etc.
        my_defaults : bool, optional, default is True
            If True, some reasonable additional defaults are set,
            namely line widths and formats.
        my_rcParams : dict, optional, default is None
            Dictionary in plt.rcParams format
            containing any other allowed global plot parameters.
    
        Returns
        -------
        None
            The result is a modification of the global plt.rcParams variable.
        '''
        # (0) Save current plot params
        # (to be eventually restored by Plots.restore_previous_params()
        Plots._previous_rcParams = plt.rcParams.copy()
        
        # (1) Basic arguments -------------------------------------------------
        if size:  # Figure size
            # Convert size in [cm] to required size in [inch]
            size = (size[0]/2.54, size[1]/2.54)
            plt.rcParams.update({'figure.figsize' : size})
        if dpi:  # Figure dpi
            plt.rcParams.update({'figure.dpi' : dpi})
        if fontsize:  # Global font size
            plt.rcParams.update({'font.size' : fontsize})
        
        # (2) Additional default parameters -----------------------------------
        if my_defaults:  # Defaults => if not forbidden by my_defaults=False
            plt.rcParams.update({
                'lines.linewidth'    : 0.8,
                'axes.linewidth'     : 0.6,
                'xtick.major.width'  : 0.6,
                'ytick.major.width'  : 0.6,
                'grid.linewidth'     : 0.6,
                'grid.linestyle'     : ':'})
        
        # (3) Further user-defined parameter in rcParams format ---------------
        if my_rcParams:  # Other possible rcParams in the form of dictionary
            plt.rcParams.update(my_rcParams)

    
    @staticmethod
    def save_plot_params():
        '''
        Save global plot parameters.
        
        * This function saves all global plot parameters (plt.rcParams).
        * The parameters can be restored by calling Plot.restore_saved_params.

        Returns
        -------
        None
            The function just saves current plot parameters.
            
        Technical note
        --------------
        The parameters are saved/copied
        to class variable Plots._saved_rcParams.
        '''
        Plots._saved_rcParams = plt.rcParams.copy()


    @staticmethod
    def restore_previous_params():
        '''
        Restore previous global plot parameters.
        
        * The global plot parameters (plt.rcParams)
          are auto-saved when calling function Plots.set_plot_params.
        * This function restores the auto-saved params
          to the previous/original state (before set_plot_params was called).

        Returns
        -------
        None
            The function just restores the previous plt.rcParams.
            
        Technical note
        --------------
        The params are saved class variable in Plots._previous_rcParams
        '''
        if Plots._previous_rcParams is not None:
            plt.rcParams.update(Plots._previous_rcParams)

    
    @staticmethod
    def restore_saved_params():
        '''
        Restore saved global plot parameters.
        
        * The global plot parameters (plt.rcParams)
          are saved when calling function Plots.save_plot_params.
        * This function restores the saved params,
          which were saved at the last call of save_plot_params function.

        Returns
        -------
        None
            The function just restores the saved plt.rcParams.
            
        Technical note
        --------------
        The params are saved class variable in Plots._saved_rcParams
        '''
  
        if Plots._saved_rcParams is not None:
            plt.rcParams.update(Plots._saved_rcParams)

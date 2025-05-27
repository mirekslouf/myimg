'''
Module: myimg.plots
-------------------

Auxiliary myimg module with functions for plotting

>>> # Simple example how to employ myimg.plots
>>> # (the module is imported and used within myimg.api
>>> import myimg.api as mi
>>> mi.plots.set_plot_parameters(size=(8,8), dpi=100)
'''

import matplotlib.pyplot as plt

def set_plot_parameters(
        size=(10,10), dpi=100, fontsize=8, my_defaults=True, my_rcParams=None):
    '''
    Set global plot parameters (this is useful for repeated plotting).

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
        containing any other allowed matplotlib parameters = rcParams.

    Returns
    -------
    None
        The result is a modification of the global plt.rcParams variable.
    '''
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
    if my_defaults:  # Default rcParams (if not my_defaults==False)
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
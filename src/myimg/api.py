'''
Module: myimg.api
------------------

A simple interface to package myimg.

>>> # Simple usage of myimg.api interface
>>> import myimage.api as mi
>>>
>>> # (1) Open image
>>> img = mi.MyImage('somefile.bmp')  # input image: somefile.bmp
>>>
>>> # (2) Modify the image 
>>> img.cut(60)                # cut off lower bar (60 pixels)             
>>> img.label('a')             # label to the upper-left corner
>>> img.scalebar('rwi,100um')  # scalebar to the lower-right corner
>>>
>>> # (3) Save the modified image 
>>> img.save_with_ext('_clm.png')  # output: somefile_ls.png

More examples are spread all over the documentation.
    
1. How to use myimg.objects:
    - myimg.objects.MyImage = single image = the basic object with many methods
    - myimg.objects.MyReport = multi-image = a rectangular grid of images
2. Specific frequent tasks:
    - myimg.objects.MyImage.scalebar = a method to insert scalebar
    - myimg.objects.MyImage.label = a method to insert label in the corner
3. Additional utilities:
    - myimg.utils = sub-package with special/more complex utilities
    - myimg.utils.scalebar = the code for myimg.objects.MyImg.scalebar method
    - myimg.utils.label = the code for myimg.objects.MyImg.label method
    - myimg.utils.fft = additional utilities, Fourier transforms
'''


import myimg.objects
import matplotlib.pyplot as plt



class MyImage(myimg.objects.MyImage):
    '''
    Class defining MyImage objects.
    
    * MyImage object = image-name + PIL-image-object + various methods.
    * This class is just inherited from myimg.objects.MyImage.
    * More help: https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html 
    '''
    pass



class MyReport(myimg.objects.MyReport):
    '''
    Class defining MyReport objects.
    
    * MyReport object = a rectangular multi-image.
    * This class is just inherited from myimg.objects.MyReport. 
    * More help: https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html 
    '''
    pass



class Apps:
    '''
    Additional applications of myimg package.
    
    * Basic features are accessible as methods of MyImage object:
    
        >>> from myimg.api import mi
        >>> img = mi.MyImage('someimage.bmp') 
        >>> img.scalebar('rwi,100um')  # basic utility, called as a method
    
    * Additional features/apps can be called as functions of Apps package:
        
        >>> from myimg.api import mi
        >>> img = mi.MyImage('someimage.bmp')
        >>> mi.Apps.fourier(img)  # additional utility, called as a function
    '''

    def FFT(img):
        pass



class Settings:
    '''
    Settings for package myimg.
    
    * This class imports all classes from myimg.settings.
    * Thanks to this import, we can use Settings myimg.api as follows:
        
    * Sample usage:
        
        >>> import myimg.api as mi
        >>> mi.Settings.Scalebar.position = (10,650)
    '''
    
    # Technical notes:
    # * All settings/defaults are in separate data module {myimg.settings};
    #   this is better and cleaner (clear separation of code and settings).
    # * In this module we define class Settings,
    #   in which we import all necessary Setting subclasses.
    # Why is it done like this?
    #   => To have an easy access to Settings for the users of this module.
    # How does it work in real life?
    #   => Import myimg.api and use Settings as shown in the docstring above.
    
    from myimg.settings import Scalebar, Label
    from myimg.settings import MicCalibrations, MicDescriptionFiles



class PlotParams:
    '''
    Simple class defining matplotlib plot parameters.
    
    In MyImg, matplotlib library is used for visualizing
    (i) input images/micrograph and
    (ii) other plots, such as histograms.
    
    * Sample usage:
        
        >>> import myimg.api as mi
        >>> mi.PlotParams.set_plot_parameters(size='8x6', dpi=100)
    '''

    
    def set_plot_parameters(
            size=(8,6), dpi=100, fontsize=8, my_defaults=True, my_rcParams=None):
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
    
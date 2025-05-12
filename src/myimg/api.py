'''
Module: myimg.api
------------------

A simple interface to package myimg.

>>> # Simple usage of myimg.api interface
>>> import myimg.api as mi
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
    - myimg.api.MyImage = single image = an image with additional methods
    - myimg.api.MyReport = multi-image = a rectangular grid of images
2. Specific frequent tasks for single images:
    - myimg.objects.MyImage.scalebar = a method to insert scalebar
    - myimg.objects.MyImage.caption = a method to add figure caption
    - myimg.objects.MyImage.label = a method to insert label in the corner
3. Additional utilities and applications:
    - myimg.utils = sub-package with code for specific/more complex methods
    - myimg.apps = sub-package with code for additional applications
    - myimg.apps.iLabels = app for immunolabelling
      (detection, classification, collocalization)
'''


import myimg.apps, myimg.objects
import matplotlib.pyplot as plt
import pandas as pd



class MyImage(myimg.objects.MyImage):
    '''
    Class providing MyImage objects.
    
    * MyImage object = PIL-image-object + image name + additional methods.
    * This class in api module (myimg.api.MyImage)
      is just inherited from objects module (myimg.objects.MyImage).
    
    >>> # Simple usage of MyImage object
    >>> import myimg.api as mi
    >>> # Open some image using MyImage class
    >>> img = mi.MyImage('somefile.png')
    >>> # Show the opened image on the screen
    >>> img.show()
    
    Parameters
    ----------
    filename : str or path-like object
        Name of the image file to work with.
    
    Returns
    -------
    MyImage object
        An image, typically after some processing (autocontrast, scalebar ...).
        MyImage objects can be shown (MyImage.show) or saved (MyImage.save).
    '''
    pass



class MyReport(myimg.objects.MyReport):
    '''
    Class providing MyReport objects.
    
    * MyReport object = a rectangular multi-image.
    * This class in api module (myimg.api.MyReport)
      is just inherited from objects module (myimg.objects.MyReport).
    
    >>> # Simple usage of MyReport object
    >>> import myimg.api as mi
    >>> # Define input images    
    >>> images = ['s1.png','s2.png']
    >>> # Combine the images into one multi-image = mreport
    >>> mrep = mi.MyReport(images, itype='gray', grid=(1,2), padding=10)
    >>> # Save the final multi-image               
    >>> mrep.save('mreport.png')   
    
    Parameters
    ----------
    images : list of images (arrays or str or path-like or MyImage objects)
        The list of images from which the MyReport will be created.
        If {images} list consists of arrays,
        we assume that these arrays are the direct input to
        skimage.util.montage method.
        If {images} list contains of strings or path-like objects,
        we assume that these are filenames of images
        that should be read as arrays.
        If {images} lists contains MyImage objecs,
        we use MyImage objects to create the final MyReport/montage.
    itype : type of images/arrays ('gray' or 'rgb' or 'rgba')
        The type of input/output images/arrays.
        If itype='gray',
        then the input/output are converted to grayscale.
        If itype='rgb' or 'rgba'
        then the input/output are treated as RGB or RGBA images/arrays.
    grid : tuple of two integers (number-of-rows, number-of-cols)
        This argument is an equivalent of
        *grid_shape* argument in skimage.util.montage function.
        It defines the number-of-rows and number-of-cols of the montage.
        Note: If grid is None, it defaults to a suitable square grid.
    padding : int; the default is 0
        This argument is an equivalent of
        *padding_width* argument in skimage.util.montage function.
        It defines the distance between the images/arrays of the montage.
    fill : str or int or tuple/list/array; the default is 'white'
        This argument is a (slightly extended) equivalent of 
        *fill* argument in skimage.util.montage function.
        It defines the color between the images/arrays.
        If fill='white' or fill='black',
        the color among the images/arrays is white or black.
        It can also be an integer value (for grayscale images)
        or a three-value tuple/list/array (for RGB images);
        in such a case, it defines the exact R,G,B color among the images.
    crop : bool; the default is True
        If crop=True, the outer padding is decreased to 1/2*padding.
        This makes the montages nicer (like the outputs from ImageMagick).
    rescale : float; the default is None
        If *rescale* is not None, then the original size
        of all input images/arrays is multiplied by *rescale*.
        Example: If *rescale*=1/2, then the origina size
        of all input images/arrays is halved (reduced by 50%).
        
    Returns
    -------
    MyReport object
        Multi-image/montage of *images*.
        MyReport objects can be shown (MyReport.show) or saved (MyReport.save).
    
    Allowed image formats
    ---------------------
    * Only 'gray', 'rgb', and 'rgba' standard formats are supported.
      If an image has some non-standard format,
      it can be read and converted using a sister MyImage class
      (methods MyImage.to_gray, MyImage.to_rgb, MyImage.to_rgba).
    * The user does not have to differentiate 'rgb' and 'rgba' images.
      It is enough to specify 'rgb' for color images
      and if the images are 'rgba', the program can handle them.
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
        '''
        Calculate FFT of img object + add it as img.FFT.

        Parameters
        ----------
        img : MyImage object
            MyImage object, created within this app.

        Returns
        -------
        None
            The result is FFT object.
            MyImage object aggregates the FFT object.
            Therefore, the FFT object is accessible as img.FFT.  
        '''
        # TODO
        pass 
    

    def iLabels(img, df=None):
        '''
        Create/read iLabels object + add it as img.iLabels.

        Parameters
        ----------
        img : MyImage object
            MyImage object, created within this app.
        df : None or Pandas.DataFrame 
            If df is a Pandas.Dataframe object,
            the iLabels are created from this object/dataframe.
            Otherwise, the iLabels are created as empty object/dataframe.

        Returns
        -------
        None
            The result is iLabels object.
            MyImage object aggregates the iLabels object.
            Therefore, the iLabels object is accessible as img.iLabels.  
        '''
        import myimg.apps.iLabels.classPeaks
        if df is None:
            img.iLabels = myimg.apps.iLabels.classPeaks.Peaks(
                img=img.img, img_name=img.name)
        elif isinstance(df, pd.DataFrame):    
            img.iLabels = myimg.apps.iLabels.classPeaks.Peaks(
                df=df, img=img.img, img_name=img.name)
        else:
            print('Error initializing MyImage.iLabels!')
            print('Wrong type of {peaks} argument!')
            print('Empty {peaks} object created.')
            img.iLabels = myimg.apps.iLabels.classPeaks.Peaks(
                img=img.img, img_name=img.name)



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
    
    from myimg.settings import Scalebar, Label, Caption
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
    
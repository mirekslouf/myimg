'''
Module: myimg.api
------------------

A simple interface to MyImg package.

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
>>> img.save_with_ext('_ls.png')  # output: somefile_ls.png

More examples are spread all over the documentation.
    
1. MyImage key objects:
    - myimg.api.MyImage = single image = an image with additional methods
    - myimg.api.MyReport = multi-image = a rectangular grid of images
2. MyImage additional applications:
    - myimg.apps = list of available applications
    - myimg.api.Apps = adding additional utils/apps to MyImage

'''


# Import modules
# --------------
# (1) Basic MyImage objects
# (myimg.objects are used within myimg.api
# >>> import myimg.api as mi        # standard myimg.api import
# >>> img = mi.MyImage('some.png')  # read image as mi.MyImage object
import myimg.objects
# (2) Auxiliary myimg module for plotting
# myimg.plots is used directly = imported to myimg.api + used for function calls
# >>> import myimg.api as mi          # standard myimg import
# >>> mi.plots.set_plot_parameters()  # direct call of myimg.plots function
import myimg.plots   # this imports plots module to myimg.api
plots = myimg.plots  # this makes it accesible as myimg.api.plots


class MyImage(myimg.objects.MyImage):
    '''
    Class providing MyImage objects.
    
    * MyImage object = PIL-image-object + additional attributes and methods.
    * This class in api module (myimg.api.MyImage)
      is just inherited from objects module (myimg.objects.MyImage).
    
    >>> import myimg.api as mi        # standard import of MyImg package
    >>> img = mi.MyImage('some.png')  # open some image
    >>> img.show()                    # show the image
    
    Parameters
    ----------
    img : image (array or str or path-like or MyImage object)
        Name of the array/image that we want to open.
    pixsize : str, optional, default is None
        Description how to determine pixel size.
        Pixel size is needed to calculate the scalebar length.
        See docs of myimg.objects.MyImage.scalebar
        about the possibilities how to use pixsize argument.
    name : str, optional, default is None
        Name of the MyImage object.
        If MyImage is created from file, the *name* is the filename.
        If MyImage is created from array, the *name* can be user-defined.
        The name is employed by MyImg.save_with_extension method.
        
    Returns
    -------
    MyImage object
        An image, which can be
        adjusted (MyImage.autocontrast, MyImage.border ...),
        processed (MyImage.label, MyImage.caption, MyImage.scalebar ...),
        shown (MyImage.show)
        or saved (MyImage.save, MyImage.save_with_extension).    
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
        Multi-image = tiled image composed of *images*.
        MyReport object can be shown (MyReport.show) or saved (MyReport.save).
    
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
    Additional applications for MyImg package.
    
    * Additional features/apps can be added using this myimg.api.Apps class.
    * More help and examples can be found in the subclasses below.
    '''
    
    @classmethod
    def import_FFT_utils(cls):
        '''
        Import Fast Fourier Transform utilities.
        
        * The function gives access to myimg.apps.fft.fft module.
        * https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg/apps.html
        
        The function can be called and used in two ways:
            
        >>> import myimg.api as mi  # ........... standard import of MyImg
        >>>
        >>> mi.Apps.import_FFT_utils()  # ....... 1st way to access fft
        >>> mi.Apps.fft.FFT('some.png')
        >>>
        >>> fft = mi.Apps.import_FFT_utils  # ... 2nd way to access fft
        >>> fft.FFT('some.png')
        '''
        # Import fft into the *local function namespace*.
        # (The module is loaded once and cached in sys.modules by Python;
        # (the name `fft` is local to this func unless we store return it.
        import myimg.apps.fft as fft
    
        # Save fft as a *class attribute*.
        # (This enables the following usage:
        # >>> import myimg.api as mi
        # >>> mi.Apps.import_FFT_utils()
        # >>> mi.Apps.fft.something(...)
        cls.fft = fft
    
        # Return the fft module.
        # (This additionally enables:
        # >>> import myimg.api as mi
        # >>> fft = mi.Apps.import_FFT_utils()
        # >>> fft.something(...)
        return fft


    @classmethod        
    def import_Velox_utils(cls):
        '''
        Import utilities for processing Velox EMD files.
        
        * The function gives access to myimg.apps.velox_util module.
        * https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg/apps.html
        
        The function can be called and used in two ways:
            
        >>> import myimg.api as mi  # ................ standard import of MyImg
        >>>
        >>> mi.Apps.import_Velox_utils()  # .......... 1st way to access Velox 
        >>> mi.Apps.Velox.EMDfiles.describe(r'./')
        >>>
        >>> Velox = mi.Apps.import_Velox_utils  # .... 2nd way to access Velox
        >>> Velox.EMDfiles.describe(r'./')
        '''
        # Import Velox into the *local function namespace*.
        # (The module is loaded once and cached in sys.modules by Python;
        # (the name `Velox` is local to this func unless we store return it.
        import myimg.apps.velox_utils as Velox

        # Save Velox as a *class attribute*.
        # (This enables the following usage:
        # >>> import myimg.api as mi
        # >>> mi.Apps.import_Velox_utils()
        # >>> mi.Apps.Velox.something(...)
        cls.Velox = Velox
        
        # Return the Velox module.
        # (This additionally enables:
        # >>> import myimg.api as mi
        # >>> Velox = mi.Apps.import_Velox_utils()
        # >>> Velox.something(...)        
        return Velox
    
    
    @classmethod
    def import_iLabels(cls):
        '''
        Import iLabels package = process STEM images with nanoparticles.

        * The function gives access to myimg.apps.iLabels package.
        * https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg/apps.html
        
        The function can be called and used in two ways:
            
        >>> import myimg.api as mi  # ............... standard import of MyImg
        >>>
        >>> mi.Apps.import_iLabels()  # ............. 1st way to access iLabels
        >>> mi.Apps.iLabels.something(...)
        >>>
        >>> iLabels = mi.Apps.import_iLabels()  # ... 2nd way to access iLabels
        >>> iLabels.something(...)
        '''
        # Import iLabels into the *local function namespace*.
        # (The package is loaded once and cached in sys.modules by Python;
        # (the name `iLabels` is local to this func unless we store return it.        
        import myimg.apps.iLabels as iLabels

        # Save iLabels as a *class attribute*.
        # (This enables the following usage:
        # >>> import myimg.api as mi
        # >>> mi.Apps.import_iLabels()
        # >>> mi.Apps.iLabels.something(...)        
        cls.iLabels = iLabels
        
        # Return the iLabels package.
        # (This additionally enables:
        # >>> import myimg.api as mi
        # >>> iLabels = mi.Apps.import_iLabels()
        # >>> iLabels.something(...)                
        return iLabels


class Settings:
    '''
    Settings for myimg package.
    
    * This class (myimg.Settings)
      imports all dataclasses from myimg.settings.
    * Thanks to this import, we can use Settings myimg.api as follows:
            
    >>> # Sample usage of Settings class
    >>> # (this is NOT a typical usage of Settings dataclasses
    >>> # (the settings are usually not changed and just used in myimg funcs
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

    


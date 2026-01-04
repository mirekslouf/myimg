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
    
1. How to use myimg.objects:
    - myimg.api.MyImage = single image = an image with additional methods
    - myimg.api.MyReport = multi-image = a rectangular grid of images
2. MyImage objects - frequent methods:
    - myimg.objects.MyImage.scalebar = a method to insert scalebar
    - myimg.objects.MyImage.caption = a method to add figure caption
    - myimg.objects.MyImage.label = a method to insert label in the corner
3. MyImage objects - additional applications:
    - myimg.api.Apps = class for adding additional utils/apps to MyImage
    - myimg.api.Apps.FFT = an example of one utility = Fourier transform
4. Additional utilities and applications:
    - myimg.plots = sub-package with auxiliary functions for plotting
    - myimg.utils = sub-package with code for specific/more complex methods
    - myimg.apps = sub-package with code for additional applications
    - myimg.apps.iLabels = app for immunolabelling
      (detection, classification, collocalization)
'''


# Import modules
# --------------
# (1) Basic MyImage objects
# (myimg.objects are used within myimg.api
# >>> import myimg.api as mi        # standard myimg.api import
# >>> img = mi.MyImage('some.png')  # read image as mi.MyImage object
import myimg.objects
# (2) Additional MyImage applications
# (additional applications can be added within myimg.api.Apps
# >>> import myimg.api as mi        # standard myimg.api import
# >>> img = mi.MyImage('some.png')  # read image as mi.MyImage object
# >>> fft = mi.Apps.FFT(img)        # create FFT of the image using mi.Apps.FFT
import myimg.apps.fft
import myimg.apps.profiles
import myimg.apps.velox
import myimg.apps.iLabels
# (3) Auxiliary myimg module for plotting
# myimg.io is used directly = imported to myimg.api + used for function calls
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
        See docs of myimg.objects.MyImage.scalebar for more details.
        
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
    * More help and examples can be found in the available applications below.
    * Links to available apps: myimg.api.Apps.FFT, myimg.api.Apps.iLabels ...
    '''


    class FFT(myimg.apps.fft.FFT):
        '''
        Class providing FFT objects.
        
        * FFT object = Fast Fourier Transform of an image/array.

        >>> # Simple usage of FFT objects
        >>> import myimg.api as mi        # standard import of myimg
        >>> img = mi.MyImage('some.png')  # open an image using myimg.api
        >>> fft = my.Apps.FFT(img)        # calculate FFT of the img object
        >>> fft.show(cmap='magma')        # show the result
        
        Parameters
        ----------
        img : image (array or str or path-like or MyImage object)
            The 2D object, from which we will calculate FFT
            = 2D-DFFT = 2D discrete fast Fourier transform.

        Returns
        -------
        FFT object.
        
        Technical details
        -----------------
        * FFT object, 3 basic attributes: FFT.fft (array of complex numbers),
          FFT.intensity (array of intensities = magnitudes = real numbers)
          and FFT.phase (array of phases = angles in range -pi:pi).
        * FFT object is pre-processed in the sense that the intensity center
          is shifted to the center of the array (using scipy.fftpack.fftshift).
        * FFT object carries the information about calibration (pixel-size),
          on condition it was created from MyImage object (the typical case).
        '''
        pass


    class RadialProfile(myimg.apps.profiles.RadialProfile):
        '''
        Class providing RadialProfile objects.
        
        * RadialProfile object = intensity distribution as a function of radius.

        >>> # Simple usage of RadialProfile objects
        >>> import myimg.api as mi
        >>> img = mi.MyImage('some.png')        # open an image
        >>> rp = mi.Apps.RadialProfile(img)     # compute radial profile
        >>> rp.show()                           # plot the result
        >>> rp.save("radial.csv")               # save profile to file
        
        Parameters
        ----------
        img : image (array or str or path-like or MyImage object)
            The 2D grayscale/binary image from which we calculate the profile.
        center : tuple of two floats, optional
            The (x, y) coordinates of the center. 
            If None, the image center is used.

        Returns
        -------
        RadialProfile object.
        
        Technical details
        -----------------
        * RadialProfile object has two main attributes:
          RadialProfile.R (array of radii in pixels),
          RadialProfile.I (array of mean intensities).
        * RadialProfile is computed by averaging pixel intensities
          in concentric circles around the center.
        * Can be visualized with `.show()` or exported with `.save()`.
        '''
        pass

    class AzimuthalProfile(myimg.apps.profiles.AzimuthalProfile):
        '''
        Class providing AzimuthalProfile objects.
        
        * AzimuthalProfile object = 
          intensity distribution as a function of angle.
    
        >>> # Simple usage of AzimuthalProfile objects
        >>> import myimg.api as mi
        >>> img = mi.MyImage('some.png')         # open an image
        >>> ap = mi.Apps.AzimuthalProfile(img)   # compute azimuthal profile
        >>> ap.show()                            # plot the result
        >>> ap.save("azimuthal.csv")             # save profile to file
        
        Parameters
        ----------
        img : image (array or str or path-like or MyImage object)
            The 2D grayscale/binary image from which we calculate the profile.
        center : tuple of two floats, optional
            The (x, y) coordinates of the center. 
            If None, the image center is used.
        bins : int, optional
            Number of angular bins (default = 360). Defines the resolution of
            the profile in degrees.
    
        Returns
        -------
        AzimuthalProfile object.
        
        Technical details
        -----------------
        * AzimuthalProfile object has two main attributes:
          AzimuthalProfile.Theta (array of angles in degrees),
          AzimuthalProfile.I (array of mean intensities).
        * AzimuthalProfile is computed by averaging pixel intensities
          in angular sectors around the center.
        * The angular range is 0–360 degrees.
        * Can be visualized with `.show()` or exported with `.save()`.
        '''
        pass


    class Velox:
        '''
        Class with utilities for Velox EMD files.
        
        >>> # Simple usage of Velox class.
        >>> import myimg.api as mi
        >>> # (1) EMDfiles class = renaming and describing EMD files.
        >>> vdir = r'd:\data.sh\velox'
        >>> mi.Apps.Velox.EMDfiles.rename(vdir)
        >>> mi.Apps.Velox.EMDfiles.describe(vdir)
        >>> # (2) EMDobject class = working with individual EMD files.
        >>> vfile = vdir + '\' + '016_h66_650kx.emd'
        >>> d = mi.Apps.Velox.EMDobject(vfile)
        >>> print(d.pixelsize())
        >>> # (3) Note: mi.Apps.Velox.EMDmetadata is usually not used directly.
        '''
        
        
        class EMDfiles(myimg.apps.velox.EMDfiles):
            '''
            EMDfiles class - rename and/or describe Velox EMD files.
            '''
            pass
        
        
        class EMDmetadata(myimg.apps.velox.EMDmetadata):
            '''
            EMDmetadata class - access to metadata of Velox EMD files.
            '''
            pass
        
        
        class EMDobject(myimg.apps.velox.EMDobject):
            '''
            Class providing EMDobjects.
            '''
            pass


    class iLabels(myimg.apps.iLabels.classPeaks.Peaks):
        '''
        Class providing iLabels objects.

        * iLabels object = peak annotation, detection, feature extraction,
          and classification for immunolabelling data.
        * Entry point: wraps `myimg.apps.iLabels.classPeaks.Peaks`.

        >>> # Simple usage of iLabels objects
        >>> import myimg.api as mi
        >>> img = mi.MyImage("annotation_12_procc.tif")
        >>> 
        >>> # Initialize iLabels with image
        >>> il = mi.Apps.iLabels(img=img.img, img_name="annotation_12_procc.tif")
        >>> 
        >>> # Load peaks (from pickle)
        >>> il.read("annot12.pkl")
        >>> 
        >>> # Show peaks
        >>> il.show_in_image()
        >>> il.show_as_text(num=5)
        >>> 
        >>> # Detect peaks automatically
        >>> il.find(method="ncc", mask_path="./masks", thr=0.2)
        >>> 
        >>> # Extract features
        >>> il.characterize(img_path="annotation_12_procc.tif",
        ...                 peak_path="annot12.pkl",
        ...                 mask_path="./masks")
        >>> 
        >>> # Classify peaks
        >>> il.classify(data=il.X_train, method="rfc", target=il.y_train)

        Parameters
        ----------
        df : pandas.DataFrame, optional
            Table with peak coordinates and labels.
        img : str or array-like or PIL.Image.Image, optional
            Input image associated with the peaks.
        img_name : str, optional
            Human-readable image name.
        file_name : str, optional
            Default filename for saving results.
        cmap : str, optional
            Colormap for display.
        messages : bool, optional
            If True, print progress messages.

        Returns
        -------
        iLabels object
            Provides methods for peak detection (`find`), feature extraction
            (`characterize`), and classification (`classify`).
        '''
        pass
        
        # OLD
        # nasledujici kod pridaval iLabels jako attribut do objektu MyImage
        # toto nakonec zavrzeno a nechano nize jen jako docasna zaloha
        # def iLabels(myimg.apps.iLabels.classPeaks.Peaks):
        #     import myimg.apps.iLabels.classPeaks
        #     if df is None:
        #         img.iLabels = myimg.apps.iLabels.classPeaks.Peaks(
        #             img=img.img, img_name=img.name)
        #     elif isinstance(df, pd.DataFrame):    
        #         img.iLabels = myimg.apps.iLabels.classPeaks.Peaks(
        #             df=df, img=img.img, img_name=img.name)
        #     else:
        #         print('Error initializing MyImage.iLabels!')
        #         print('Wrong type of {peaks} argument!')
        #         print('Empty {peaks} object created.')
        #         img.iLabels = myimg.apps.iLabels.classPeaks.Peaks(
        #             img=img.img, img_name=img.name)


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

    


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
>>> img.save_with_ext('_clm.png')  # output: somefile_clm.png
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


class MyImage:
    '''
    Class defining MyImage objects.
    
    * MyImage object = image name + PIL image object + extra props/methods.
    * See __init__ for more information about initial object parameters.
    '''
        
    def __init__(self, filename):
        '''
        Initialize MyImage object.

        Parameters
        ----------
        filename : str or path-like object
            Name of the image file to work with.

        Returns
        -------
        MyImage object
            MyImage object contains:
            (i) name of the original image (MyImage.name),
            (ii) corresponding PIL image object  (MyImage.img), and
            (iii) further properties and methods (MyImage.cut, crop, ...).
        '''
        self.name = filename
        self.img  = MyImage.open_image(filename)
        self.width, self.height = self.img.size
        
    
    @staticmethod
    def open_image(filename):
        '''
        Open image file using PIL.Image taking into account all exceptions.

        Parameters
        ----------
        filename : str
            Name of the file to open.

        Returns
        -------
        img : PIL image object
            The PIL image object is usually saved in MyImage object.
        '''
        try:
            img = Image.open(filename)
            return(img)
        except FileNotFoundError:
            print(f'File not found: {filename}')
            sys.exit()
        except IOError as err:
            print(f'Error opening image: {filename}')
            print(err)
            sys.exit()
        except OSError as err:
            print(f'OS error when opening: {filename}')
            print(err)
            sys.exit()
    
    
    def get_font_size(self, font_name, required_font_size_in_pixels):
        '''
        Get font size (in fontsize units)
        corresponding to *required_font_size_in_pixels*.

        Parameters
        ----------
        fontname : str
            Name of the TrueType font to use.
            Example (working in Windows): font_name='timesbd.ttf'
        required_font_size_in_pixels : float
            Required font size in pixels.

        Returns
        -------
        font_size : int
            Final font size (in font units).
            If the returned *font_size* (in fontsize units)
            is applied to font with given *fontname*,
            then the height of the font (in pixels units)
            will correspond to *required_font_size_in_pixels* argument.
            
        Technical notes
        ---------------
        * This function is a modified recipe from StackOverflow:
          https://stackoverflow.com/q/4902198
        * My modification may be a bit slower (not too much)
          but it seems to be clear and reliable.
        '''
        # (1) Initial fontsize = required_font_size
        # (Note: fontsize [in fontsize units]
        # (is just APPROXIMATELY equal to required fontsize [in pixels].
        font_size = round(required_font_size_in_pixels)
        
        # (2) Initialize draw_object + font_object and calculate font height
        # (Note1: draw_object is needed for the correct font height calculation
        # (Note2: we calculate the font height for a model text - here: cap 'M'
        draw_object = ImageDraw.Draw(self.img)
        font_object = ImageFont.truetype(font_name, font_size)
        font_height = MyImage.font_height_pix(draw_object, font_object) 
        # (2a) Current font_height > required_font_size_in_pixels, decrease...
        while font_height > required_font_size_in_pixels:
            font_size -= 1
            font_object = ImageFont.truetype(font_name, font_size)
            font_height = MyImage.font_height_pix(draw_object, font_object)
        # (2b) Current font_height > required_font_size_in_pixels, increase...
        while font_height < required_font_size_in_pixels:
            font_size += 1
            font_object = ImageFont.truetype(font_name, font_size)
            font_height = MyImage.font_height_pix(draw_object, font_object)
        
        # (3) Return font size (for given font_name, in font_name units)
        return(font_size)
    
    
    @staticmethod
    def font_height_pix(draw_object, font_object):
        bbox = draw_object.textbbox((20, 20), 'M', font=font_object)
        text_height = bbox[3] - bbox[1]
        return(text_height)
    
    
    def cut(self, height_of_bar):
        '''
        Cut off lower bar with given height.

        Parameters
        ----------
        height_of_bar : int
            Height of the lower bar to cut.
            Lower bars are typical of many microscopic images.
            A lower bar contains information from given microscope,
            but it is usually removed when preparing the image for publication.

        Returns
        -------
        None
            The output is saved in self.img.
        '''
        # Cut off lower bar
        self.img = self.img.crop(
            (0,0, self.width, self.height - height_of_bar))
        # Update image size
        self.width, self.height = self.img.size
    
    
    def crop(self, rectangle):
        '''
        Crop image = keep just selected rectangular area.

        Parameters
        ----------
        rectangle : tuple of four integers
            Tuple (X1,Y1,X2,Y2),
            where X1,Y1 = coordinates of upper left corner
            and X2,Y2 = coordinates of lower right corner.

        Returns
        -------
        None
            The output is saved in *self.img*.
        '''
        # Crop image
        self.img = self.img.crop(rectangle)
        # Update image size
        self.width, self.height = self.img.size
    
                
    def label(self, label, F=None, **kwargs):
        '''
        Insert a one-letter label in the upper left corner of an image. 

        Parameters
        ----------
        label : str
            One letter label that will be inserted in the upper left corner.
        F : float, optional, default is None
            Multiplication coefficient/factor that changes the label size.
            If F = 1.2, then all label parameters are enlarged 1.2 times.
        kwargs : list of keyword arguments
            Allowed keyword arguments are:
            color, bcolor, position, stripes, messages.
            See section *List of allowed kwargs* for detailed descriptions.
            
        Returns
        -------
        None
            The label is drawn directly to *self.img*.

        List of allowed kwargs
        ----------------------
        * color : PIL color specification, default is 'black'.
            Text color = color of the label text.
        * bcolor : PIL color specification, default is 'white'.
            Background color = color of the label background/box.

        Technical notes
        ---------------
        * Transparent background:
          To set transparent background,
          set optional/keyword argument bcolor='transparent'.
          It is not enough to omit bcolor,
          because all omitted keyword arguments
          are set to their defaults defined in Settings.Label.
          In the case of omitted bcolor argument, the default is 'white'. 
        * Color label in grayscale image:
          To set color label in grayscale image,
          it is necessary to convert image to RGB;
          otherwise the colored label would be converted to grayscale.
        '''
        
        # The complete code of this method is long. 
        # Therefore, the code has been moved to its own module.
        # This method is a wrapper calling the function in the external module. 
        from myimg.utils import label as my_label
        my_label.insert_label(self, label, F, **kwargs)
        
                
    def scalebar(self, pixsize, F=None, **kwargs):
        '''
        Insert a scalebar in the lower right corner of the image.

        Parameters
        ----------
        pixsize : str
            Description how to determine pixel size.
            Pixel size is needed to calculate the scalebar length.
            See *Example* section below to see available options.
        F : float, optional, the default is None
            Multiplication coefficient/factor that changes the scalebar size.
            If F = 1.2, then all scalebar parameters are enlarged 1.2 times.
        kwargs : list of keyword arguments
            Allowed keyword arguments are:
            color, bcolor, position, stripes, messages.
            See section *List of allowed kwargs* for detailed descriptions. 
            
        Returns
        -------
        None
            The scalebar is drawn directly to *self.img*.
            
        List of allowed kwargs
        ----------------------
        * color : PIL color specification, default is 'black'.
            Text color = color of the scalebar text and line.
        * bcolor : PIL color specification, default is 'white'.
            Background color = color of the scalebar background/box.
        * length : str, default is None.
            If length is given (using a string such as '100um','1.2nm')
            then the lenght fixed at given value and not calculated by the
            program (calculation would yield some reasonable lenght of
            scalebar around 1/6 of the image width; this default is saved
            in myimg.settings.Scalebar.length property - which can be changed).
        * position : list or tuple or None, default is None.
            If position = None, the scalebar is drawn
            at the default position in the lower-right corner of the image.
            If position = (X,Y) = list or tuple of two integers,
            the scalebar is drawn at position X,Y of the image.
        * stripes : bool or int, default is False.
            If stripes = False, draw standard scalebar.
            If stripes = True or 1, draw scalebar with 5 stripes.
            If stripes = N, where N>=2, draw striped scalebar with N stripes.
        * messages : bool, default is False.
            If messages=True, print info about program run.
        
        Example
        -------
        >>> # Four basic ways how to insert a scalebar in an image
        >>> # (model example; in real life we use just one of the ways
        >>>
        >>> # (0) Import api + read image
        >>> import myimage.api as mi
        >>> img = mi.MyImage('../IMG/image123_20kx.bmp')
        >>> 
        >>> # (1) Pixel size from real width of image = 100um
        >>> img.scalebar('rwi,100um')
        >>>
        >>> # (2) Pixel size from a known length in image => 50 nm = 202 pixels
        >>> img.scalebar('knl,50nm,202')
        >>> 
        >>> # (3) Pixel size from known magnification
        >>> # (note: this can be done only for calibrated microscope
        >>> # (calibrated microscopes => myimg.settings.MicCalibrations
        >>> # (3a) magnification deduced from last part of image name
        >>> # (note: mag = everything between last underscore and suffix
        >>> # (in this example we have: ../IMG/image123_20kx.bmp => mag = 20kx
        >>> img.scalebar('mag,TecnaiVeleta')
        >>> # (3b) magnification inserted directly
        >>> # (note: mag can be something like 20kx, 20k, 20000x, 20000
        >>> img.scalebar('mag,TecnaiVeleta,20kx')
        >>>
        >>> # (4) Pixel size from accompanying text file
        >>> # (note: some microscopes save images + descriptive txt files
        >>> # (the format of text file must be described somehow
        >>> # (description of text files => myimg.settings.MicDescriptionFiles
        >>> img.scalebar('txt,MAIA3')
        >>> 

        Technical notes
        ---------------
        * Transparent background:
          To set transparent background,
          set optional/keyword argument bcolor='transparent'.
          It is not enough to omit bcolor,
          because all omitted keyword arguments
          are set to their defaults defined in Settings.Scalebar.
          In the case of omitted bcolor argument, the default is 'white'. 
        * Color scalebar in grayscale image:
          To set color label in grayscale image,
          it is necessary to convert image to RGB;
          otherwise the colored label would be converted to grayscale.
        '''
        
        # The complete code of this method is long. 
        # Therefore, the code has been moved to its own module.
        # This method is a wrapper calling the function in the external module. 
        from myimg.utils import scalebar as my_scalebar
        my_scalebar.insert_scalebar(self, pixsize, F, **kwargs)
    
    
    def show(self, cmap=None, axes=False):
        '''
        Show image.

        Parameters
        ----------
        cmap : matplotlib colormap name, optional, default is None
            Matplotlib colormap name.
            If omitted and we have grayscale image, then we use cmap=gray.
        axes : bool, optional, default is False
            If axes=False (default), do not show axes around the image.

        Returns
        -------
        None
            The output is the image shown on the screen.

        '''
        arr = np.asarray(self.img)
        if arr.ndim == 2: cmap='gray'
        if not(axes): plt.axis('off')
        plt.imshow(arr, cmap=cmap)
        plt.show()
    
    
    def save(self, output_image):
        '''
        Save image using arbitrary output dir, name and extension.

        Parameters
        ----------
        output_image : str or path-like object
            Filename of the output image.
            The format of saved image is guessed from the extension.

        Returns
        -------
        None
            The output is the saved *output_image*.
        '''
        self.img.save(output_image)
    
    
    def save_with_extension(self, my_extension):
        '''
        Save image in the same dir with slightly modified name and extension.

        Parameters
        ----------
        my_extension : str
            Specific extension of the output image.
            The argument my_extension can extend image name
            + modify image extension/format;
            see the example below.
            
        Returns
        -------
        None
            The output is the saved output image with *my_extension*.
        
        Example
        -------
        >>> import myimage.api as mi
        >>> # Open image
        >>> img = mi.MyImage('../IMG/somefile.bmp')
        >>> # Cut off lower bar 
        >>> img.cut(60)
        >>> # Save the image as: '../IMG/somefile_cut.png')
        >>> img.save_with_ext('_cut.png')
        '''
        (file,ext) = os.path.splitext(self.name)
        output_image = file + my_extension
        self.img.save(output_image)
            

class Utils:
    '''
    Additional utilities of myimg package.
    
    * Basic utilities are accessible as methods of MyImage object:
    
        >>> from myimg.api import mi
        >>> img = mi.MyImage('someimage.bmp') 
        >>> img.scalebar('rwi,100um')  # basic utility, called as a method
    
    * Additional utilities can be called as functions of Utils package:
        
        >>> from myimg.api import mi
        >>> img = mi.MyImage('someimage.bmp')
        >>> mi.Utils.fourier(img)  # additional utility, called as a function
    '''

    
    def montage(imgs, tile, **kwargs):
        pass


    def fourier(img):
        pass


class Settings:
    '''
    Settings for package myimg.
    
    * This class just imports all classes from myimg.settings.
    * Reason: In order to get acces to these classes within api package.
    * Conclusion: thanks to this import, we can do thinks like:
        
        >>> import myimg.api as mi
        >>> mi.Settings.Scalebar.position = (10,650)
    '''
    
    # Technical notes:
    # 0) All settings/defaults are in separate data module {myimg.settings};
    #    this is better and cleaner (clear separation of code and settings).
    # 1) At the very beginning of the whole module {myimg.api} we have:
    #    import myimg.settings as Settings 
    # 2) Here = in class {myimg.api.Settings} we have:
    #    from myimg.settings import Scalebar, Label ...
    # Why is it done this way + how does it work?
    # ad 1) Convenient access to Settings in this whole module and its classes:
    #       i.e. anywhere inside this module we can do: Settings.Scalebar ....
    # ad 2) Convenient access to Settings for users of the module, such as:
    #       import myimg.api as mi
    #       mi.Settings.Scalebar ...
    
    from myimg.settings import Scalebar, Label
    from myimg.settings import MicCalibrations, MicDescriptionFiles

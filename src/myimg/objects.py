'''
Module: myimg.objects
---------------------

Key classes/objects for myimg package:

1. *MyImage* class
   defines the basic MyImg object,
   which is used in most image manipulations.
2. *Montage* class
   defines a set/montage of images,
   which are arranged in a rectangular tiled image.
3. *Units*, *NumberWithUnits* and *ScaleWithUnits* classes
   defines allowed units, number-with-units and scale-with-units, respectively.

Examples = how does it work?
----------------------------

**MyImage** class creates the basic object,
which is used in most image manipulations within myimg.api module.

>>> # MyImage class :: simple example (short but real)
>>> import myimg.api as mi        # import MyImage API ~ simple UI
>>> img = mi.MyImage('some.bmp')  # open image: some.bmp
>>> img.label('a')                # insert a label in the upper left corner
>>> img.scalebar('rwi,100um')     # insert a scalebar to the lower right corner
>>> img.save_with_ext('_s.png')   # save the modified image to: some_ls.png

**MyReport** class creates the multi-image object,
which contains several images arranged in a rectangular grid.

>>> # MyReport class :: simple example (short but real)
>>> import myimg.api as mi        # import MyImage API ~ simple UI
>>> images = ['s1.png','s2.png']  # define input images
>>> mrep = mi.MyReport(images,    # create montage image
>>>     itype='gray', grid=(1,2), # ...grayscale, just two images in a row
>>>     padding=10)               # ...padding/spacing between imgs = 10pixels
>>> mrep.save('mreport.png')      # save the final montage of (the two) images   

**Units**, **NumberWithUnits**, and **ScaleWithUnits** classes
are used in myimg.utils.scalebar module
when creating scalebars (as a scalebar contains *number with units*).
More information can be found below at the definitions of
myimg.objects.Units, myimg.objects.NumberWithUnits,
and myimg.objects.ScaleWithUnits.
'''

import os, sys, re, pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps
import skimage as ski
from dataclasses import dataclass


class MyImage:
    '''
    Class defining MyImage objects.    
    '''


    def __init__(self, img, pixsize=None):
        '''
        Initialize MyImage object.

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
        '''
        
        # (1) Open the image and define/save its name
        # (we have to consider the type of the input img argument
        if type(img) in (str, pathlib.WindowsPath, pathlib.PosixPath):
            # (a) img is str or path-like object = filename
            # => open the file to PIL.Image and save its filename
            self.img = MyImage.img_from_file(img)
            self.name = img
        elif type(img) == np.ndrarray:
            # (b) img is a numpy array
            # => convert the file to PIL.Image and create profisional name
            self.img = MyImage.img_from_array(img)
            self.name = 'image_from_array'
        elif type(img) == MyImage:
            # (c) img is an existing MyImage object
            # => create a copy of the object
            self.img = img.img
            self.name = img.name
        else:
            print('Error initializing MyImage object!')
            print(f'Unsuported type of input: f{type(img)}')
            sys.exit()
            
        # (2) Define/determine additional image parameters
        # (again, we have to considet the type of the input img argument
        if type(img) == MyImage:
            # (a) MyImage object
            # => copy all relevant parameters
            self.width     = img.width
            self.height    = img.height
            self.itype     = img.itype
            self.pixsize = img.pixsize
        else:
            # (b) array or filename
            # => determine the basic parameters
            self.width, self.height = self.img.size
            self.itype = self.set_image_type()
            self.pixsize = None
        
        # (3) Redefine pixsize attribute if requested
        # (pixel size can be defined
        # (during initialization of MyImage
        # (by means of optional pixsize argument
        if pixsize is not None: self.set_scale(pixsize)
        
    
    @staticmethod
    def img_from_file(filename):
        '''
        Create PIL.Image object from an image file.

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
            print(f'IO error when opening image: {filename}')
            print(err)
            sys.exit()
        except OSError as err:
            print(f'OS error when opening: {filename}')
            print(err)
            sys.exit()
    
    
    @staticmethod
    def img_from_array(arr):
        '''
        Create PIL.Image object from an array.

        Parameters
        ----------
        arr : numpy array
            Name of the array that should be opened as PIL.Image object.

        Returns
        -------
        img : PIL image object
            The PIL image object is usually saved in MyImage object.
        '''
        try:
            img = Image.from_array(arr)
            return(img)
        except Exception as err:
            print('Something went wrong when converting array to PIL.Image!')
            print(err)
            sys.exit()
            
        
    def set_image_type(self):
        '''
        Set the image type.
        
        This method gets the image type
        AND converts some less common image type to more standard ones.

        Returns
        -------
        Image type
            We use image types: 'binary', 'gray', 'gray16', 'rgb', 'rgba'.
        '''
        # Get the image type
        # AND convert some less common image types to more standard.
        imode = self.img.mode
        if imode == '1':
            itype = 'binary'
        elif imode == 'L':
            itype ='gray'
        elif imode == 'P':
            self.img = self.img.convert('L')
            itype ='gray'
        elif imode == 'RGB':
            itype = 'rgb'
        elif imode == 'RGBA':
            itype = 'rgba'
        elif imode == 'I;16':
            itype = 'gray16'
        else:
            print('Unsuported image type!')
            print(f'PIL image mode: {imode}')
            print('End of program.')
            sys.exit()
        # Return the image type
        return(itype)
    
    
    def set_scale(self, pixsize):
        '''
        Set scale = define the pixel size.

        Parameters
        ----------
        pixsize : str
            Description how to determine pixel size.
            Pixel size is needed to calculate the scalebar length.
            See docs of myimg.objects.MyImage.scalebar for more details.

        Returns
        -------
        None
            The result is the defined pixel size,
            saved in the attribute pixsize of myimg.objects.MyImage object.
        '''  
        # The complete code of this method is long. 
        # Therefore, the code has been moved to its own module.
        # This method is a wrapper calling the function in the external module. 
        from myimg.utils import scalebar as my_scalebar
        self.pixsize = my_scalebar.get_pixel_size(self, pixsize)


    def set_font_size(self, font_name, required_font_size_in_pixels):
        '''
        Set font size (in fontsize units)
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
        * The function is employed in external functions
          which insert label and/or scalebar to MyImage object.
        '''
        
        # (0) Define auxiliary function to get font height
        # (the font height is calculated from fictive textbox with 'M' letter
        def font_height_pix(draw_object, font_object):
            bbox = draw_object.textbbox((20, 20), 'M', font=font_object)
            text_height = bbox[3] - bbox[1]
            return(text_height)
        
        # (1) Initial fontsize = required_font_size
        # (Note: fontsize [in fontsize units]
        # (is just APPROXIMATELY equal to required fontsize [in pixels].
        font_size = round(required_font_size_in_pixels)
        
        # (2) Initialize draw_object + font_object and calculate font height
        # (Note1: draw_object is needed for the correct font height calculation
        # (Note2: we calculate the font height for a model text - here: cap 'M'
        draw_object = ImageDraw.Draw(self.img)
        font_object = ImageFont.truetype(font_name, font_size)
        font_height = font_height_pix(draw_object, font_object) 
        # (2a) Current font_height > required_font_size_in_pixels, decrease...
        while font_height > required_font_size_in_pixels:
            font_size -= 1
            font_object = ImageFont.truetype(font_name, font_size)
            font_height = font_height_pix(draw_object, font_object)
        # (2b) Current font_height > required_font_size_in_pixels, increase...
        while font_height < required_font_size_in_pixels:
            font_size += 1
            font_object = ImageFont.truetype(font_name, font_size)
            font_height = font_height_pix(draw_object, font_object)
        
        # (3) Return font size (for given font_name, in font_name units)
        return(font_size)

                    
    def to_gray(self, itype='8bit'):
        '''
        Convert image to grayscale.

        Parameters
        ----------
        itype : str, optional, default is '8bit'
            The image is converted to 8bit grayscale.
            Only the standard 8bit grayscale images are supported now.

        Returns
        -------
        None
            The output is saved in self.img.
            
        Technical notes
        ---------------
        * We fully support only 8-bit grayscale images;
          the same situation is in Pillow (full support only for 8-bit gray).
        * Working with 16-bit grayscale images is surprisingly tricky.
          You *can* read and work with 16-bit images,
          but before using some methods of this package
          (such as label, scalebar, montage),
          they *should* be converted to 8-bit grayscale
          in order to avoid errors or strange results.
        * This method can convert
          the standard 16-bit grayscale to the 8-bit grayscale,
          but it does not support other, less common grayscale formats.
        * The less common grayscale formats can be normalized and converted
          to 8-bit grayscale manually, in an analogous way as in this method.
        '''
        if itype == '8bit':
            # Conversion to 8-bit grayscale
            if self.img.mode in ('RGB','RGBA'):
                # Conversion of RGB and RGBA images should be fine.
                self.img = self.img.convert('L')
                self.itype = 'gray'
            elif self.img.mode == 'L':
                # The image is 8-bit grayscale - just reset itype='gray'.
                self.itype='gray'
            elif self.img.mode == 'I;16':
                # Conversion of 16-bit grayscale to 8-bit grayscale
                # (requires special treatment, not supported by Pillow.
                arr = np.array(self.img)
                normalized = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                normalized = normalized.astype(np.uint8)
                self.img = Image.fromarray(normalized)
                self.itype = 'gray'
            else:
                print('Conversion to grayscale - unsupported image type!')
        else:
            # Conversion to non-8bit grayscale formats not supported now.
            print('Conversion to grayscale - only 8-bit grayscale supported!')
            print('The original image was not changed.')
            
    
    def to_rgb(self, itype='24bit'):
        '''
        Convert image to RGB.

        Parameters
        ----------
        itype : str, optional, default is '24bit'
            The image is converted to standard RGB image = 24bit = 3*8bit.
            Only the standard 24bit RGB images are supported at the moment.

        Returns
        -------
        None
            Te output is saved in self.img.
            
        Technical notes
        ---------------
        * Standard RGB format = 24bit RGB = 8 bits for each of (R,G,B) values.
        * Other (non-standard) RGB formats are not supported at the moment.
        * RGB formats: https://en.wikipedia.org -> RGB color formats
        * Pillow: https://pillow.readthedocs.io -> Handbook - Concepts - Modes
        '''
        if itype == '24bit':
            # Conversion to 24-bit RGB.
            self.img = self.img.convert('RGB')
        else:
            # Conversion to non-standard RGB formats not supported.
            print('Only standard 24-bit RGB images are supported.')
            print('The original image was not changed.')


    def to_rgba(self, itype='32bit'):
        '''
        Convert image to RGBA.

        Parameters
        ----------
        itype : str, optional, default is '32bit'
            The image is converted to standard RGBA image = 32bit = 3*8+8bit.
            Only the standard 32bit RGBA images are supported at the moment.

        Returns
        -------
        None
            Te output is saved in self.img.
            
        Technical notes
        ---------------
        * Standard RGBA format = 32bit RGBA = 8bits (R,G,B) + 8bit alpha.
        * Other (non-standard) RGB formats are not supported at the moment.
        * RGB formats: https://en.wikipedia.org -> RGB color formats
        * Pillow: https://pillow.readthedocs.io -> Handbook - Concepts - Modes
        '''
        if itype == '32bit':
            # Conversion to 24-bit RGB.
            self.img = self.img.convert('RGBA')
        else:
            # Conversion to non-standard RGBA formats not supported.
            print('Unknown image type when converting to RGBA!')
            print('The original image was not changed.')


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
    
    
    def resize(self, width=None, height=None, resample=None):
        '''
        Resize image to new width or height, keeping the aspect ratio.
    
        Parameters
        ----------
        width : int, optional, default is None
            The required width of the resized image.
            The height will be calculated
            so that the aspect ratio was preserved.
        height : int, optional, default is None
            The required height of the resized image.
            The width will be calculated
            so that the aspect ratio was preserved.
        resample : int or None
            If None, default resampling filter is used (usually Ok).
            It is also possible to define a specific resampling filter.
            For sharp images or plots we may prefer no resampling and 
            we can use *resampling = 0 = Image.Resampling.NEAREST*.
            More: https://pillow.readthedocs.io -> Reference -> Image -> resize
            
        Returns
        -------
        None
            The output is saved in *self.img*
        '''
        # (0) Calculate AR = aspect_ratio
        aspect_ratio = self.height/self.width
        
        # (1) Calculate new dimensions while keeping aspect_ratio
        if width is not None:     # New dimensions if the width is given
            new_width = round(width)
            new_height = round(width * aspect_ratio)
        elif height is not None:  # New dimensions if the height is given
            new_height = round(height)
            new_width  = round(height * 1/aspect_ratio)
        else:
            print('Image resize: width or height was not given - no action!')
            return
        
        # (2) Update image dimensions
        self.width  = new_width
        self.height = new_height 
        
        # (2) Resize the image
        self.img = self.img.resize(
            (new_width, new_height), resample=resample)

    
    def autocontrast(self, **kwargs):
        '''
        Enhance image contrast (using PIL.ImageOps.autocontrast function).

        Parameters
        ----------
        **kwargs : dict
            Any arguments of PIL.ImageOps.autocontrast function.

        Returns
        -------
        None
            The modified image is saved in self.img object.
        '''
        self.img = ImageOps.autocontrast(image = self.img, **kwargs)

        
    def border(self, border=1, color='black'):
        '''
        Draw a border around an image. 

        Parameters
        ----------
        border : int or tuple, optional, default is 1
            Int = the same thickness of border around all four edges.
            Tuple of 2 ints = (left/righ and top/bottom) border sizes.
            Tuple of 4 ints = (left, top, right, bottom) border sizes. 
        color : PIL color specification, default is 'black'
            A short text that will be inserted at the bottom of an image.
        '''        
        # (0) In the future, we may add border with shadow
        # Now we add just simple border using PIL.ImageOps.expand.
        
        # (1) Add border
        self.img = ImageOps.expand(self.img, border=border, fill=color)
        
        # (2) Update MyImage properties = width and height
        self.width  = self.img.size[0]
        self.height = self.img.size[1]

                 
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
            The default is defined in myimg.settings.Caption
            (and that is why it does not have to be re-defined here).
        * bcolor : PIL color specification, default is 'white'.
            Background color = color of the label background/box.
            The default is defined in myimg.settings.Caption
            (and that is why it does not have to be re-defined here).

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

         
    def caption(self, text, F=None, **kwargs):
        '''
        Insert a one-line textual description at the bottom of an image. 

        Parameters
        ----------
        text : str
            A short text that will be inserted at the bottom of an image.
        F : float, optional, default is None
            Multiplication coefficient/factor that changes the text size.
            If F = 1.2, then all label parameters are enlarged 1.2 times.
        kwargs : list of keyword arguments
            Allowed keyword arguments are:
            color, bcolor, messages.
            See section *List of allowed kwargs* for detailed descriptions.
            
        Returns
        -------
        None
            The label is drawn directly to *self.img*.

        List of allowed kwargs
        ----------------------
        * color : PIL color specification, default is 'black'.
            Text color = color of the label text.
            The default is defined in myimg.settings.Caption
            (and that is why it does not have to be re-defined here).
        * bcolor : PIL color specification, default is 'white'.
            Background color = color of the label background/box.
            The default is defined in myimg.settings.Caption
            (and that is why it does not have to be re-defined here).
        * align : int or str
            This parameter determines the alignment of the figure caption.
            If align = integer,
            x_position of the caption is {align} pixels from left.
            If align = string,
            it can be either 'left' or 'Left' or ''

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
        from myimg.utils import caption as my_caption
        my_caption.insert_caption(self, text, F, **kwargs)
    

    def scalebar(self, pixsize=None, F=None, **kwargs):
        '''
        Insert a scalebar in the lower right corner of the image.

        Parameters
        ----------
        pixsize : str, optional, default is None
            Description how to determine pixel size.
            Pixel size is needed to calculate the scalebar length.
            See *Example* section below to see available options.
            If pixsize is None, it is taken from self.pixsize.
            If pixsize is None and self.pixsize is not defined - error, exit.
        F : float, optional, the default is None
            Multiplication coefficient/factor that changes the scalebar size.
            If F = 1.2, then all scalebar parameters are enlarged 1.2 times.
        kwargs : list of keyword arguments
            Allowed keyword arguments are:
            color, bcolor, position, stripes, messages.
            See section *List of allowed kwargs* below for more info. 
            
        Returns
        -------
        None
            The scalebar is drawn directly to *self.img*.
            
        Examples
        --------
        
        Example 1 :: Four basic options HOW to define pixelsize + scalebar
        
        >>> # Example 1 :: 4 basic options HOW to define pixsize + scalebar
        >>> import myimg.api as mi
        >>> img = mi.MyImage('some.png')
        >>> # Option 1: real-width-of-image
        >>> # (here: rwi = 100 micrometers
        >>> img.scalebar('rwi,100um')
        >>> # Option 2: known-length-in-image
        >>> # (here: known length of 100 nm = 220 pixels        
        >>> img.scalebar('knl,100nm,220')
        >>> # Option 3: calibrated microscope + magnification
        >>> # (here: microscope LM_Nikon1 + magnification 20x     
        >>> img.scalebar('mag,LM_Nikon1,20x')
        >>> # Option 4: microscope with description files
        >>> # (here: microscope MAIA3, which gives micrographs with BHD-files
        >>> img.scalebar('txt,MAIA3)
        
        Example 2 :: Three basic ways WHEN we can define pixelsize + scalebar
                         
        >>> # Example 2 :: 3 basic ways WHEN we can define pixsize + scalebar
        >>> import myimg.api as mi
        >>> # 1st way: when we insert the scalebar
        >>> img = mi.MyImage('some.png')
        >>> img.scalebar('rwi,100um')
        >>> # 2nd way: when we read the input image
        >>> img = mi.MyImage('some.png', pixsize='rwi,100um')
        >>> img.scalebar()
        >>> # 3rd way: after reading of the input image, using set_scale
        >>> img = mi.MyImage('some.png')
        >>> img.set_scale('rwi,100um')
        >>> img.scalebar()
        
        Example 3 :: Four basic ways how to insert a scalebar in more detail

        >>> # Example 3 :: 4 basic ways HOW to insert a scalebar IN MORE DETAIL
        >>> # (this is a model example; in real life, we use just one way)
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
        >>>
        >>> # (3a) magnification deduced from last part of image name
        >>> # (note: mag = everything between last underscore and suffix
        >>> # (in this example we have: ../IMG/image123_20kx.bmp => mag = 20kx
        >>> img.scalebar('mag,TecnaiVeleta')
        >>>
        >>> # (3b) magnification inserted directly
        >>> # (note: mag can be something like 20kx, 20k, 20000x, 20000
        >>> img.scalebar('mag,TecnaiVeleta,20kx')
        >>>
        >>> # (4) Pixel size from accompanying text file
        >>> # (note: some microscopes save images + descriptive txt files
        >>> # (the format of text file must be described somehow
        >>> # (description of text files => myimg.settings.MicDescriptionFiles
        >>> img.scalebar('txt,MAIA3')

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
    
    
    def show(self, cmap=None, axes=False, iscale=False):
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
        # Grayscale images require special treatment
        grayscale_image = self.itype == 'gray' or self.itype == 'gray16'
        if grayscale_image:
            # If not specified explicitly, cmap should be gray.
            if cmap == None: cmap='gray'
            # If not required, grayscale images should not be autoscaled.
            if iscale == False:
                # ...limits for non-intensity-autoscaled grayscale-8bit
                if self.itype == 'gray': imin,imax = (0,255)
                # ...limits for non-intensity-autoscaled grayscale-16bit
                if self.itype == 'gray16': imin,imax = (0,65535)
        # Do not show axes if axes = False
        if not(axes): plt.axis('off')
        # Plot the image
        # ...intensity of grayscale images should not be autoscaled
        if grayscale_image and iscale == False:
            plt.imshow(self.img, cmap=cmap, vmin=imin, vmax=imax)
        else:
            plt.imshow(self.img, cmap=cmap)
        # Show the plot 
        plt.show()
    
    
    def save(self, output_image, dpi=300):
        '''
        Save image using arbitrary output dir, name and extension.

        Parameters
        ----------
        output_image : str or path-like object
            Filename of the output image.
            The format of saved image is guessed from the extension.
        dpi : int, optional, default is 300
            The dpi of the saved image.

        Returns
        -------
        None
            The output is the saved *output_image*.
        '''
        self.img.save(output_image, dpi=(dpi,dpi))
    
    
    def save_with_extension(self, my_extension, dpi=300):
        '''
        Save image in the same dir with slightly modified name and extension.

        Parameters
        ----------
        my_extension : str
            Specific extension of the output image.
            The argument my_extension can extend image name
            + modify image extension/format;
            see the example below.
        dpi : int, optional, default is 300
            The dpi of the saved image.
            
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
        self.img.save(output_image, dpi=(dpi,dpi))

    
class MyReport:
    '''
    Class defining MyReport objects.
    '''
    
    
    def __init__(self, images, itype,
                 grid=None, padding=0, fill='white', crop=True, rescale=None):
        '''
        Initialize MyReport object.

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
            Example: If *rescale* = 1/2, then the original size
            of all input images/arrays is halved (reduced by 50%).
            
        Returns
        -------
        MyReport object
            Multi-image = tiled image composed of *images*.
            The MyReport object can be shown (MyReport.show)
            or saved (MyReport.save).
        
        Technical notes
        ---------------
        * Only 'gray', 'rgb', and 'rgba' standard formats are supported.
          If an image has some non-standard format,
          it can be read and converted using a sister MyImage class
          (methods MyImage.to_gray, MyImage.to_rgb, MyImage.to_rgba).
        * The user does not have to differentiate 'rgb' and 'rgba' images.
          It is enough to specify 'rgb' for color images
          and if the images are 'rgba', the program can work with them.
        '''
        
        # Get basic arguments/properties
        self.images  = images
        self.itype   = itype
        self.grid    = grid
        self.padding = padding
        self.fill    = fill
        self.crop    = crop
        self.rescale = rescale
        
        # Process self.images
        # 1) Convert all images to np.arrays.
        # 2) Check if the np.arrays have the same size,
        #    i.e. the same dimensions, including the number of channels.
        #    (channels ~ colors + transparency; usually the last dim/axis
        # 3) Rescale images/arrays if requested.
        #    (the images/arrays are rescaled if self.rescale != None
        self._process_images()
        
        # Process self.itypes
        # 1) Check image types
        #    (the only allowed image types are 'rgb' and 'gray'.
        # 2) Define additional parameters for montage:
        #    (the additional parameters will be saved within the self object
        self._check_image_types()
        
        # Rescale images if requested
        # (rescaleing can be performed AFTER...
        # (...self.process_images => because images are converted to arrays
        # (...self.check_image_types => because this sets additional params:
        # (   self.montage_channel_axis
        # (   self.montage_number_of_channels
        if self.rescale is not None: self._rescale_images()
        
        # Adjust white/black colors according to image type and no-of-channels
        if (self.itype == 'rgb') or (self.itype == 'rgba'): 
            image_black = (0,)   * self.montage_number_of_channels
            image_white = (255,) * self.montage_number_of_channels 
        elif self.itype == 'gray':
            image_black = 0
            image_white = 255
        else:
            # TODO: Add support for binary images.
            print('MyReport: error during initialization!')
            print('Unknown/unsupported image type.')
            sys.exit()
                
        # Analyze fill = fill_color and adjust acc.to image_type
        match fill:
            case 'white':
                self.fill = image_white
            case 'black':
                self.fill = image_black
    
        # Create final montage
        self.montage = ski.util.montage(
            arr_in=self.images,
            grid_shape=self.grid, fill=self.fill,
            padding_width=self.padding, channel_axis=self.montage_channel_axis)
        
        # Crop the montage (= reduce the outer padding size to 1/2*padding)
        if self.padding and self.crop:
            my_crop = round(self.padding / 2)
            self.montage = self.montage[my_crop:-my_crop, my_crop:-my_crop]

    
    def _process_images(self):
        '''
        Check and prepare images before creating MyReport/montage.
        
       * The images used for MyReport/montage
         should be convertible to arrays with the same type and size.
       * This method goes through all images in self.images
         and converts them to arrays.
       * If the converted images/arrays have the same dimensions(=type),
         then everything is Ok
         and we cen proceed.
       * If the conerted images/arrays do not have the same dimension(=type),
         then this methods ends with an error
         and the user has to adjust images BEFORE calling the montage method.

        Parameters
        ----------
        None
            Auxiliary method of MyReport object; the only argument is self.
        
        Returns
        -------
        None
            If everything is Ok,
            the checked images are read and saved as arrays in the self.images.
            If the images are not of the same type
            this method prints an error and the program quits.
        '''
        
        # (1) Check dimensions of all images.
        
        # (1a) Prepare empty list for image dimensions
        img_sizes = []
        
        # (1b) Go through the images (self.images) and save their dimensions 
        for i,image in enumerate(self.images):
            # Case 1: NumPy array => just append its dimension = array.shape 
            if type(image) == np.ndarray:
                # self.images is not changed = we already have an array
                # just save image size
                img_sizes.append(image.shape)
            # Case 2: MyImage object => convert to array and append 
            elif 'MyImage' in str(type(image)):
                # self.images[i] is changed => np.array instead of MyImg object
                self.images[i] = np.array(image.img)
                # save image size
                img_sizes.append(self.images[i].shape)
            # Case 3: None object => append None
            # TODO: Later, None should be exchanged for suitable empty array.
            elif type(image) == None:
                pass
            # Case 4: Nothing of the above => we suppose it is a filename.
            # TODO: Consider exceptions etc...
            else:
                # self.images[i] is changed => we read a file to an array
                if 'rgb' in self.itype:
                    # Read image in the default way (as_gray=False)
                    self.images[i] = ski.io.imread(image, as_gray=False)
                elif 'gray' in self.itype:
                    # Read image as grayscale (as_grayscale=True)
                    self.images[i] = ski.io.imread(image, as_gray=True)
                else:
                    print('Error in: myimg.Montage.process_images')
                    print('Unknown input image type!')
                    sys.exit()
                # Save image size = append it to img_sizes list
                img_sizes.append(self.images[i].shape)
        
        # (1c) Check if all image sizes are the same
        # Trick (a bit thorny, but verified): All-list-elements-are-the-same
        #  if the number of the occurrences of the 1st (or 2nd, 3rd..) element
        #  equals to the total number of elements in the list.
        the_same_dimensions = img_sizes.count(img_sizes[0]) == len(img_sizes)
        # Now the final check
        if not(the_same_dimensions):
            print('MyReport: error during initialization!')
            print('The images do not have the same dimensions.')
            sys.exit()
                
        # (2) Additional adjustment for all grayscale images
        # * if itype = 'gray', we should normalize image to 8bit grayscale
        # * in ski.io.imread, the grayscale images may have different formats:
        #   => 3-value gray-imgs normalized to 1, 8bit to 255, 16bit to 65535
        # * moreover, we can have a combination of various grayscales
        #   => to get reproducible results, we ALWAYS normalize to 8-bit gray
        for i,image in enumerate(self.images):
            if self.itype == 'gray':
                arr_max = np.max(self.images[i])
                self.images[i] = np.round(
                    self.images[i]/arr_max * 255).astype(np.uint8)
        
        # (3) If all tests/adjustments above were Ok,
        # we can return the list of adjusted images in the form of arrays.
        return(self.images)
    
    
    def _check_image_types(self):
        '''
        Check image/array types and prepare additional parametrs for montage.
        
        * The only allowed image types are 'rgb' and 'gray'.
        * Other image types => print error message and exit program.
        * For each image type we have to set two additional montage params:
            - self.montage_channel_axis => the last axis of the array
            - self.montage_number_of_channels => gray ~ 1, rgb ~ 3, rgba ~ 4
        
        Parameters
        ----------
        None
            Auxiliary method of MyReport object; the only argument is self.
        
        Returns
        -------
        None
            The additional montage parameters are are saved in
            self.montage_channel_axes and self.montage_number_of_channels.
            These parameters are employed within MyReport object
            for correct treatement of rgb/gray images/arrays.
        '''
        
        if (self.itype == 'rgb') or (self.itype == 'rgba'):
            # RGB and RGBA images = 3D-arrays
            # the last axis = channel axis = color channels = axis 4 => index 3
            # the number of channels = either 3 (RGB) or 4 (RGBA)
            self.montage_channel_axis = 3
            self.montage_number_of_channels = self.images[0].shape[-1]
        elif self.itype == 'gray':
            # Grayscale images = 2D-arrays
            # (no channel axis = just one channel, grayscale value
            # (all grayscale images are read by ski.io.imread as 1-value gray
            self.montage_channel_axis = None
            self.montage_number_of_channels = 1
        else:
            # Other image formats are not supported at the moment.
            print(f'Unknown image type: [{self.itype}]!')
            print("Allowed image types are 'gray', 'rgb' or 'rgba'.")
            sys.exit()
            
    
    def _rescale_images(self):
        '''
        Rescale input images/arrays if requested.
        
        * The images are rescaled if self.rescale != None.
        * This rescaling should be applied to each of self.images
          AFTER the images have been checked and converted to arrays.
       
        Parameters
        ----------
        None
            Auxiliary method of MyReport object; the only argument is self.
       
        Returns
        -------
        None
            The rescaled images/arrays are saved in self.images.
        '''
        
        # Rescale all input images/arrays if requested.
        if self.rescale is not None:
            # Go through the images/arrays in arr_in
            # and rescale them one-by-one,
            # preserving the range and data type.
            for i,image in enumerate(self.images):
                # Save the original data type.
                original_data_type = image.dtype
                # Prepare channel axis
                if self.montage_channel_axis is not None:
                    channel_axis = self.montage_channel_axis - 1
                else:
                    channel_axis = None
                # Rescale the array, preserving the range and data type.
                self.images[i] = ski.transform.rescale(
                    image,                        # input montage array
                    self.rescale,                 # rescaling coefficient
                    anti_aliasing=True,           # anti-aliasing
                    preserve_range=True,          # preserve intensities
                    channel_axis=channel_axis     # channel axis for RGB images
                    ).astype(original_data_type)  # preserve original data type


    def to_gray(self):
        '''
        Convert MyReport to grayscale image.
        
        Parameters
        ----------
        None
            In fact, this method is rarely needed.
            The MyReport image type (gray, rgb) is defined at initialization.
            Nevertheless, we *might have used* color images for montage,
            but we may want to convert the final report to grayscale.

        Returns
        -------
        None
            The MyReport is just converted
            so that it can be saved as a grayscale image.
        '''
        
        # (1) Auxiliary function to convert RGB images to grayscale.
        # https://stackoverflow.com/q/12201577
        def rgb2gray(arr):
            return( np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140]) )
        
        # (2) Main code of the method
        # (convert image to grayscale, considering known image types
        # (the image type should be saved in MyReport.itype = self.itype
        if self.itype == 'gray':
            pass
        elif self.itype in ('rgb','rgba'):    
            self.montage = rgb2gray(self.montage)
            self.itype = 'gray'
        else:
            # TODO: Support for binary images.
            print('MyReport: error when converting to grayscale!')
            print('Unknown/unsupported image type.')
            sys.exit()
    
        
    def show(self, cmap=None, axes=False):
        '''
        Show MyReport = rectangular montage of images.

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
            The output is the MyReport (image montage) shown on the screen.

        '''
        # Check/adjust the colormap
        # (if we have grayscale image and no colormap, we want cmap='gray'
        if (cmap == None) and (self.itype == 'gray'): cmap = 'gray'
        # Prepare the plot
        plt.imshow(self.montage, cmap=cmap)
        # Switch off the axes, if not requested otherwise
        if not(axes == True): plt.axis('off')
        # Show the plot
        plt.show()
    
    
    def save(self, output_image, dpi=300):
        '''
        Save MyReport (using an arbitrary path, name and extension).

        Parameters
        ----------
        output_image : str or path-like object
            Filename of the output image.
            The format of saved image is guessed from the extension.
        dpi : int, optional, default is 300
            The dpi of the saved image/montage.

        Returns
        -------
        None
            The output is the saved *output_image*.
        '''
        # Convert to grayscale, if required.
        if self.itype == 'gray':
            self.montage = (
                self.montage/np.max(self.montage) * 255).astype(np.uint8)
        # Save the output image montage.
        # (ski.io.imsave does not support saving with defined DPI
        # (plt.imsave supports DPI, but grayscale images require cmap='gray'
        if self.itype == 'gray':
            # Grayscale images are saved with colormap = 'gray'
            plt.imsave(output_image, self.montage, dpi=dpi, cmap='gray')
        else:
            # RGB (and RGBA) images are saved without cmap to keep the colors
            # Note: if we use plt.imsave (instead of ski.io.imsave) ...
            # ... we must convert array to contiguous so that it worked
            # ... in most cases it works anyway, but not always (mystery)
            arr_contiguous = np.ascontiguousarray(self.montage)
            plt.imsave(output_image, arr_contiguous, dpi=dpi)
        
               
@dataclass
class Units:
    '''
    Data class: just a container for the following two dataclasses.
    
    * The sub-dataclasses are used in NumberWithUnits/ScaleWithUnits objects.
    * The units definitions are fixed, the users should not change them.
    * The correct units (Lenghts or RecLenghts) are guessed
      during NumberWithUnits/ScaleWithUnits object initialization.
      Example: If `nwu = NumberWithUnits('2um')`, then we use `Units.Lenghts`.
    '''

    
    @dataclass
    class Lengths:
        '''
        Data class: length units (and their ratios) for micrographs.    
        '''
        micro      : str = '\u00B5'
        angstrem   : str = '\u212B'
        micrometer : str = micro + 'm'
        units      : tuple = ('m','cm','mm', 'um', 'nm', 'A')
        ratios     : tuple = ( 1, 100, 1000, 1e6, 1e9, 1e10)

    
    @dataclass
    class RecLengths:
        '''
        Data class: reciprocal lenght units (and their ratios).
        
        * Reminder: reciprocal lenghts are used in diffractograms
        * Here: rust a pair of units that are employed in real life: 1/nm, 1/A
        '''
        angstrem      : str = '\u212B'
        rec_angstrem  : str =  '1/'+angstrem
        units         : tuple = ('1/nm','1/A')
        ratios        : tuple = ( 1, 10)


class NumberWithUnits:
    '''
    Class defining NumberWithUnits object.
    
    * NumberWithUnits object = number + units.
    * The numbers-with-units are used for pixel sizes or scalebars.
      
    Object initialiation
    
    >>> # Three basic ways how to initialize a NumberWithUnits object
    >>> # (this is a NON-typical usage
    >>> # (NumberWithUnits object is used internally, when drawing scalebars
    >>> from myimg.nwu import NumberWithUnits
    >>> nwu1 = NumberWithUnits('1.2um')
    >>> nwu2 = NumberWithUnits(number=1.2, units='um')
    >>> nwu3 = NumberWithUnits(a_number_with_units_object_such_as_nwu1)
    
    List of object properties
        
    * number = number/numeric value
    * units = units corresponding to number
    * _units_description = myimg.settings.Units subclass, private property
    
    Object methods
    
    * text = return number-with-units as string
    * units_Ok = test if the defined units are correct
    * index_of_units = index of units in units_description.units list
    * increase_units = increase units and modify number accordingly
    * decrease_units = decrease units and modify number accordingly
    * set_units_to = set units to given units and modify number accordingly
    * set_correct_units = set units so that the number was within <1,1000)
    '''

    
    def __init__(self, number=None, units=None):
        '''
        Initialize NumberWithUnits object.

        Parameters
        ----------
        number : float or str or NumberWithUnits object
            Number with (or without) units, which can be:
            (i) Number (float; such as: 100),
            but then the 2nd argument should be given (such as units='um'). 
            (ii) String (str; such as: '100 um' or '1.2nm').
            (iii) Another NumberWithUnits object;
            in such a case we receive the copy of the argument.
              
        units : str, optional, the default is None
            If the 1st argument (number) is a float, the 2nd argument (units)
            defines units of the first argument.
            If the 1st argument is string or NumberWithUnits object,
            the 2nd argument is ignored.
            
        Returns
        -------
        NumberWithUnits object
            NumberWithUnits object contains:
            (i) numerical value (NumberWithUnits.number),
            (ii) corresponding units (NumberWithUnits.units), and
            (iii) further props/methods (NumberWithUnits.change_units ...).
        '''
        # (1) Determine input types
        if number is not None: input1 = type(number)
        if units  is not None: input2 = type(units)
        # (2) Parse the input number with units
        if input1 == float or input1 == int:
            # 1st input is a float (number) => 2nd input should be str (units)
            if input2 == str:
                self.number = number
                self.units  = units
            else:
                print('Error in defining NumberWithUnits object!')
                print('Units not given, exiting...')
                sys.exit()
        elif input1 == str:
            # input is string => split using regexp
            # TODO: consider errors and exceptions
            m = re.search(pattern=r'''
                # 1st/2nd group = number/units in standard () => to remember
                # exponent/rec.value in (?:) => not to remember
                (                  # start of 1st group = number
                  \d*\.?\d*        # number, such as 1, 1.2, .9
                  (?:e[+-]?\d+)?   # optional exponent, such as e3 e-6
                )                  # end of 1st group
                \s*                # possible whitespace
                (                  # start of 2nd group = units
                  (?:1/)?          # optional reciprocal value = '1/'
                  \D+              # one or more non-digit characters
                )                  # end of 2nd group
                ''',
                string = number,
                flags = re.VERBOSE)
            number,units = m.groups()
            self.number = float(number)
            self.units  = units
        elif input1 == NumberWithUnits:
            # input is another NumberWithUnits => create copy of it
            self.number = number.number
            self.units  = number.units
        else:
            print('Error in defining NumberWithUnits:')
            print('Wrong input number_with_units!')
            sys.exit()
        # (3) Determine, which units we use
        # => set private _units_description property
        if self.units in Units.Lengths.units:
            self._units_description = Units.Lengths
        elif self.units in Units.RecLengths.units:
            self._units_description = Units.RecLengths
        else:
            print('Error in defining NumberWithUnits:')
            print('Uknown units!')
            sys.exit()
    
    def __str__(self):
        '''Overwritten __str__ method to get nice output with print.'''
        # Get number and units (they will be adjusted, originals unchanged).
        number = self.number
        units  = self.units
        # Adjust number: if it is (very close to) integer, convert to integer.
        eps = 1e-10
        if abs(number - int(number)) < eps: number = int(round(number))
        # Adjust units: if units contains a special character, print it.
        if   units == 'um' : units = self._units_description.micrometer
        elif units == 'A'  : units = self._units_description.angstrem
        elif units == '1/A': units = self._units_description.rec_angstrem
        # Combine final number and units
        text = str(number) + ' ' + units
        # Return resulting string
        return(text)

    
    def text(self):
        '''
        Return number + units as string.

        Returns
        -------
        str : number-with-units
            The method returns the saved number with units as string.
            Example: if self.number = 1.2 and self.units = nm, we get '1 nm'.
            The units are printed in unicode (important for um and agstrems).
        '''
        return(self.__str__())

    
    def units_Ok(self, units=None):
        '''
        Test if current units are correct.

        Parameters
        ----------
        units : str, optional, the default is None
            If units='something', then 'something' is compared
            with the list of allowed units for self = NumberWithUnits object.

        Returns
        -------
        bool
            True if the units are correct and False otherwise.
        '''
        # (1) If units argument is not given,
        # check units of current NumberWithUnits object.
        if units is None:
            units = self.units
        # (2) Return True/False if units are correct/incorrect.
        if units in self._units_description.units:
            return(True)
        else:
            return(False)

        
    def index_of_units(self, units_to_check=None):
        '''
        Get index of current or specified units.

        Parameters
        ----------
        units_to_check : str, optional, default is None
            Any string representing some units.
            If units_to_check is not given,
            the method check current units of self object.
            
        Returns
        -------
        int
            Index of current or specified units.
            If units = 'um' then index_of_units = 2,
            because Units.Lenghts.units = ('m', 'mm', 'um', 'nm', 'A'),
            which means: 'm' => 0, 'mm' => 1, 'um' => 2...
        
        Note
        ----
        This function is employed in further functions
        manipulating with units, such as increase_units, decrease_units...
        '''
        if units_to_check == None: units_to_check = self.units 
        index_of_units = self._units_description.units.index(units_to_check)
        return(index_of_units)

        
    def increase_units(self):
        '''
        Increase current units (for example: um -> mm).

        Returns
        -------
        None
            The result is saved in NumberWithUnits object.
        '''
        # (1) Get index of current units.
        i = self.index_of_units()
        # (2) If index-of-curent-units = 0,
        # then it is not possible to increase units - we are at maximum. 
        if i == 0:
            print('It is not possible to increase units!')
        # (3) If index-of-current-units > 0,
        # we can: 1) calculate ratio, 2) decrease number, 3) increase units.
        else:
            ratio_between_units = (
                self._units_description.ratios[i] /
                self._units_description.ratios[i-1] )
            self.number = self.number / ratio_between_units
            self.units  = self._units_description.units[i-1]


    def decrease_units(self):
        '''
        Decrease current units (for example: um -> nm).

        Returns
        -------
        None
            The result is saved in NumberWithUnits object.
        '''
        # (1) Get index of current units.
        i = self.index_of_units()
        max_index = len(self._units_description.units) - 1
        # (2) If index-of-curent-units = last-index-of-our-units,
        # then it is not possible to decrease units - we are at minimum. 
        if i == max_index:
            print('It is not possible to decrease units!')
        # (3) If index-of-current-units < last-index-of-our-units,
        # we can: 1) calculate ratio, 2) increase number, 3) decrease units.
        else:
            ratio_between_units = (
                self._units_description.ratios[i] /
                self._units_description.ratios[i+1])
            self.number = self.number / ratio_between_units
            self.units  = self._units_description.units[i+1]

        
    def set_units_to(self, target_units):
        '''
        Set units to *target_units* and modify number accordingly.

        Parameters
        ----------
        target_units : str
            Any string specifying valid units.

        Returns
        -------
        None
            The results = changed units (and correspondingly changed number)
            are saved in NumberWithUnits object.
        '''
        index_current    = self.index_of_units()
        index_target     = self.index_of_units(target_units)
        index_difference = index_target - index_current
        if index_difference > 0:
            for i in range(index_difference): self.decrease_units()
        elif index_difference < 0:
            for i in range(abs(index_difference)): self.increase_units()

                
    def set_correct_units(self):
        '''
        Set correct units, so that the number was between 1 and 1000.

        Returns
        -------
        None
            The result is saved in NumberWithUnits object.
        '''
        while self.number < 1:
            self.decrease_units()
        while self.number >= 1000:
            self.increase_units()


class ScaleWithUnits(NumberWithUnits):
    '''
    Class defining ScaleWithUnits object.
    
    * ScaleWithUnits object = number + units + pixels.
    * The objet defines a scalebar:
      its lenght (number,units) and length-in-pixels (pixels).
    
    Object initialiation
    
    >>> # Three basic ways how to initialize a ScaleWithUnits object
    >>> # (this is a NON-typical usage
    >>> # (ScaleWithUnits object is used internally, when drawing scalebars
    >>> from myimg.nwu import ScaleWithUnits
    >>> swu1 = ScaleWithUnits('1.2um', pixels=100)
    >>> swu2 = ScaleWithUnits(number=1.2, units='um', pixels=100)
    >>> swu3 = ScaleWithUnits(a_number_with_units_object, pixels=100)
    
    List of object properties
        
    * number = number/numeric value
    * units  = units corresponding to number
    * pixels = pixels corresponding to number-with-units
    * _units_description: myimg.settings.Units subclass, private property
    
    Object methods
    
    * inherited methods from myimg.nwu.NumberWithUnits
    * adjust_lenght_to = adjust lenght in pixels and modify number accordingly
    * adjust_scalebar_size = adjusts scalebar lenght to some reasonable size
    '''

    
    def __init__(self, number=None, units=None, pixels=None):
        '''
        Initialize ScaleWithUnits object.

        Parameters
        ----------
        number : float or str or NumberWithUnits object
            Number with (or without) units, which can be:
            (i) Number (float; such as: 100),
            but then the 2nd argument should be given (such as units='um'). 
            (ii) String (str; such as: '100 um' or '1.2nm').
            (iii) Another NumberWithUnits object;
            in such a case we receive the copy of the argument.
        units : str, optional, the default is None
            If the 1st argument (number) is a float, the 2nd argument (units)
            defines units of the first argument.
            If the 1st argument is string or NumberWithUnits object,
            the 2nd argument is ignored.
        pixels : float
            Length of scalebar in pixels.
            This is a keyword argument that is formally optional, but in fact 
            it must be specified so that the initialization made sense.
            Moreover, this argument must be specified as keyword argument,
            because the 2nd argument (units) is really optional and
            it may not be clear, which argument is which.
            
        Returns
        -------
        ScaleWithUnits object
            NumberWithUnits object contains:
            (i) numerical value (NumberWithUnits.number),
            (ii) corresponding units (NumberWithUnits.units),
            (iii) length-of-scalebar-in-pixels and
            (iv) further props/methods (most of which
            are inherited from NumberWithUnits superclass).
        '''
        super().__init__(number, units)
        self.pixels = pixels

    
    def __str__(self):
        '''Overwritten __str__ method to get nice output with print.'''
        text = \
            'Scalebar: ' + super().__str__() + \
            ', length-in-pixels: ' + str(self.pixels)
        return(text)

    
    def text(self):
        '''
        Return number-with-units as string (ignoring pixels).

        Returns
        -------
        str : number-with-units
            The method returns the saved number with units as string.
            Example: if self.number = 1.2 and self.units = nm, we get '1 nm'.
            The units are printed in unicode (important for um and agstrems).

        '''
        # We need JUST text.
        # => therefore, we call super().__str__() - this returns just text.
        # => if we called self.__str__(), we would get text + pixel-size-info.
        text = super().__str__()
        return(text)

    
    def adjust_length_to(self, n):
        '''
        Set lenght-of-scalebar to *n* and modify lenght-in-pixels accordingly.

        Parameters
        ----------
        n : float
            The new length-of-scalebar (= self.number).

        Returns
        -------
        None
            The new lenght-of-scalebar and lenght-of-scalebar-in-pixels
            are saved in ScaleWithUnits.number and ScaleWithUnits.pixels
            properties, respectively.
        '''
        self.pixels = self.pixels * n/self.number
        self.number = n

        
    def adjust_scalebar_size(self):
        '''
        Set scalebar to some reasonable lenght
        and modify the lenght-in-pixels accordingly.

        Returns
        -------
        None
            The modified (number, units, and pixels) are saved in
            ScaleWithUnits object.
            
        Examples
        --------
        * swu = 0.3 um => swu.adjust_scalebar_size() => 300 nm
        * swu = 2.3 um => swu.adjust_scalebar_size() => 2 um
        * swu = 456 um => swu.adjust_scalebar_size() => 500 um
        * swu = 888 um => swu.adjust_scalebar_size() => 1 mm
        '''
        # Get the number in the interval <1...1000)
        self.set_correct_units()
        # Adjust the number to a reasonable value (1,2,3,5,10,20,30...),
        # modify the lenght-in-pixels accordingly, and change units if needed.
        if   self.number > 750: self.adjust_length_to(1000)
        elif self.number > 350: self.adjust_length_to( 500)
        elif self.number > 250: self.adjust_length_to( 300)
        elif self.number > 150: self.adjust_length_to( 200)
        elif self.number >  50: self.adjust_length_to( 100)
        elif self.number >  35: self.adjust_length_to(  50)
        elif self.number >  25: self.adjust_length_to(  30)
        elif self.number >  15: self.adjust_length_to(  20)
        elif self.number > 7.5: self.adjust_length_to(  10)
        elif self.number > 3.5: self.adjust_length_to(   5)
        elif self.number > 2.5: self.adjust_length_to(   3)
        elif self.number > 1.5: self.adjust_length_to(   2)
        else                  : self.adjust_length_to(   1)
        # Correct units in case we achieved the value of 1000.
        self.set_correct_units()
        # Round number (for nicer output: 1 um instead of 1.0 um)
        self.number = round(self.number)
        # Round pixels (for correct drawing in image = whole pixels)
        self.pixels = round(self.pixels)

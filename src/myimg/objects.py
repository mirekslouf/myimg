'''
Module: myimg.objects
---------------------

Key classes/objects for myimg package:

1. *MyImg* class
   defines the basic MyImg object,
   which is used in most image manipulations.
2. *Montage* class
   defines a montage/combination of several images,
   which are arranged in a rectangular grid/multi-image.
3. *Units, NumberWithUnits and ScaleWithUnits* classes
   defines allowed units, number-with-units and scale-with-units, respectively.

How does it work?
-----------------

*MyImg* class creates the very basic object,
which is used in most image manipulations within myimg.api module.

>>> # MyImg class :: simple example (short but real)
>>> import myimage.api as mi      # import API, which provides simple UI
>>> img = mi.MyImage('some.bmp')  # open image: some.bmp
>>> img.label('a')                # insert label in upper left corner
>>> img.scalebar('rwi,100um')     # insert scalebar to lower right corner
>>> img.save_with_ext('_s.png')   # save modified image: to some_ls.png

*Montage* class creates the multi-image object,
which contains several images arranged in a rectangular grid.

>>> # Montage class :: simple example (short but real)
>>> import myimage.api as mi      # import API, which provides simple UI
>>> images = ['s1.png','s2.png']  # define input images
>>> montage = mi.Montage(images,  # create montage image
>>>     itype='gray', grid=(1,2), # ...grayscale, just two images in a row
>>>     padding=10)               # ...padding/spacing between imgs = 10pixels
>>> montage.save('mreport.png')   # save the final montage of (the two) images   

*Units, NumberWithUnits, and ScaleWithUnits* classes
are used in myimg.utils.scalebar module
when creating scalebars (as a scalebar contains *number with units*).

>>> # Units, NumberWithUnits and ScaleWithUnits classes :: typical/indirect use 
>>> # (the classes are employed behind the scenes when calling scalebar method
>>> img = mi.MyImage('some.bmp')  # open image: some.bmp
>>> img.scalebar('rwi,100um')     # scalebar (employs Units, NumberWithUnits... 
>>> img.save_with_ext('_s.png')   # save modified image: to some_s.png
   
>>> # NumberWithUnits class :: direct use (non-typical, but possible)
>>> from myimg.objects import NumberWithUnits
>>> n =NumberWithUnits('0.1mm')
>>> print('Initial number_with_units:', n)  # prints 0.1 mm
>>> print('Setting correct units...')
>>> n.set_correct_units()
>>> print(n)                                # prints 100 um
'''


import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from dataclasses import dataclass
import skimage as ski


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


class Montage:
    '''
    Class defining Montage objects.
    
    * Montage object = a rectangular multi-image. 
    * See __init__ for more information about initial object parameters.
    '''
    
    def __init__(self, images, itype,
                 grid=None, padding=0, fill='white', crop=True, rescale=None):
        '''
        Initialize Montage object.

        Parameters
        ----------
        images : list of images (str or path-like objects or arrays)
            The list of images from which the montage will be created.
            If {images} list consists of strings or path-like objects,
            we assume that these are filenames of images
            that should be read as arrays.
            If {images} list consists of arrays,
            we assume that these are the direct input to montage
            (more precisely: the direct input to skimage.util.montage method).
        itype : type of images/arrays ('gray' or 'rgb')
            The type of input/output images/arrays.
            If itype='gray',
            then the input/output are converted to grayscale.
            If itype='rgb',
            then the input/output are treated as RGB images/arrays.
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
        MyImage object
            MyImage object contains:
            (i) name of the original image (MyImage.name),
            (ii) corresponding PIL image object  (MyImage.img), and
            (iii) further properties and methods (MyImage.cut, crop, ...).
        '''
        
        # Get basic arguments/properties
        self.images  = images
        self.itype   = itype
        self.grid    = grid
        self.padding = padding
        self.fill    = fill
        self.crop    = crop
        self.rescale = rescale
        
        # Analyze itype = image_type and define additional parameters
        match itype:
            case 'gray' :
                image_black = 0
                image_white = 1
                self.montage_channel_axis = None
            case 'rgb':
                image_black = (0,0,0)
                image_white = (255,255,255)
                self.montage_channel_axis = 3
            case _:
                print('Unknown image type!')
                sys.exit()
        
        # Analyze fill = fill_color and adjust acc.to image_type
        match fill:
            case 'white':
                self.fill = image_white
            case 'black':
                self.fill = image_black
        
        # Go through the images and prepare the montage image/array.
        montage_array = self.prepare_montage_array()
                    
        # Create final montage
        self.montage = ski.util.montage(
            arr_in=montage_array,
            grid_shape=self.grid, fill=self.fill,
            padding_width=self.padding, channel_axis=self.montage_channel_axis)
        
        # Crop the montage (= reduce the outer padding size to 1/2*padding)
        if self.padding and self.crop:
            my_crop = round(self.padding / 2)
            self.montage = self.montage[my_crop:-my_crop, my_crop:-my_crop]


    def prepare_montage_array(self):
        '''
        Prepare array for montage.
        
        * This function is usually not used directly.
        * It is called during the initialization of Montage object. 

        Returns
        -------
        arr_in : NumPy array
            An input array of images/arrays for creating the final montage.
        '''
        
        # (0) Prepare empty array for montage = montage array.
        arr_in = []
        
        # (1) Go through the images (self.images) to create montage array.
        for image in self.images:
            # Case 1: NumPy array => just append
            if type(image) == np.ndarray:
                arr_in.append(image)
            # Case 2: MyImage object => convert to array and append 
            elif type(image) == MyImage:
                arr_in.append(np.array(image.img))
            # Case 3: None object => append None
            # TODO: Later, None should be exchanged for suitable empty array.
            elif type(image) == None:
                arr_in.append(None)
            # Case 4: Nothing of the above => we suppose it is a filename.
            # TODO: Consider exceptions etc...
            else:
                if 'rgb' in self.itype:
                    arr_in.append(ski.io.imread(image, as_gray=False))
                else:
                    arr_in.append(ski.io.imread(image, as_gray=True))
                    # We have to differentiate three cases
                    # according to maximal intensity in the image.
                    # (a) Get the maximal intensity of the last array.
                    max_intensity = np.max(arr_in[-1])
                    # MaxIntensity = 1 => RGB image or 3-value grayscale
                    # (the most common => skimage rescales intensities to (0,1)
                    # (=> white color is 1
                    if max_intensity == 1:
                        self.fill = 1
                    # MaxIntensity <= 255 => True/1-value/8-bit grayscale
                    # (not so frequent => skimage does not rescale intensities
                    elif max_intensity <= 255:
                        self.fill = 255
                    # MaxIntensity <= 65535 => True/1-value/16-bit grayscale
                    # (not so frequent => skimage does not rescale intensities
                    elif max_intensity <= 65535:
                        self.fill = 65535
                        
        # (2) Resize all input images/arrays if requested.
        if self.rescale is not None:
            # Go through the images/arrays in arr_in
            # and rescale theym one-by-one,
            # preserving the range and data type.
            for i in range(len(arr_in)):
                # Save the original data type.
                original_data_type = arr_in[i].dtype
                # Prepare channel axis
                if self.montage_channel_axis is not None:
                    channel_axis = self.montage_channel_axis - 1
                else:
                    channel_axis = None
                # Rescale the array, preserving the range and data type.
                arr_in[i] = ski.transform.rescale(
                    arr_in[i],                    # input montage array
                    self.rescale,                 # rescaling coefficient
                    anti_aliasing=True,           # anti-aliasing
                    preserve_range=True,          # preserve intensities
                    channel_axis=channel_axis     # channel axis for RGB images
                    ).astype(original_data_type)  # preserve original data type
        
        # (3) Return the montage array
        return(arr_in)

    def show(self, cmap=None, axes=False):
        '''
        Show image montage.

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
            The output is the image montage shown on the screen.

        ''' 
        ski.io.imshow(self.montage)
        ski.io.show()
    
    
    def save(self, output_image):
        '''
        Save image montage (using an arbitrary path, name and extension).

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
        # Convert to grayscale, if required.
        if self.itype == 'gray':
            self.montage = (
                self.montage/np.max(self.montage) * 255).astype(np.uint8)
        # Save the output image montage.
        ski.io.imsave(output_image, self.montage)

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
    >>> from myimg.nwu import NumberWithUnits
    >>> nwu1 = ScaleWithUnits('1.2um')
    >>> nwu2 = ScaleWithUnits(number=1.2, units='um')
    >>> nwu3 = ScaleWithUnits(number_with_units_object_such_as_nwu1)
    
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
        if abs(number - int(number)) < eps: number = int(number)
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
    >>> from myimg.nwu import ScaleWithUnits
    >>> swu1 = ScaleWithUnits('1.2um', pixels=100)
    >>> swu2 = ScaleWithUnits(number=1.2, units='um', pixels=100)
    >>> swu3 = ScaleWithUnits(number_with_units_object, pixels=100)
    
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
        and modify lenght-in-pixels accordingly.

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
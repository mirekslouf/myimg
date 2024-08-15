'''
Module: myimg.utils.scalebar
----------------------------

This module defines function insert_scalebar, with the following features:

* The function inserts a scalebar into a micrograph (or diffractogram).
* The function employs many auxiliary functions defined this module.
* The function is usually not called directly, but through myimg.api:
    
>>> # Inserting scalebar using myimg.api interface
>>> import myimage.api as mi
>>>
>>> img = mi.MyImage('somefile.bmp')
>>>
>>> # img.scalebar calls our myimg.utils.scalebar.insert_scalebar function
>>> img.scalebar('rwi,100um')
>>>
>>> img.save_with_ext('_clm.png')

Documentation of the functions in this module:

* The functions do not have doctstings as they are not used directly.
* Nevertheless, the functions are reasonably documented in the source code.
* In pdoc-generated-HTML, click *View source code* to get the commented code.

For documentation of *insert_scalebar* function in this module, see docs
of the calling myimg.api.MyImage.scalebar method
(the usaage of the scalebar method is shown in the example above).
'''

import sys, os, glob, re
import numpy as np
from PIL import ImageFont, ImageDraw
import myimg.settings as Settings
from myimg.objects import NumberWithUnits, ScaleWithUnits
 
           
def insert_scalebar(my_img, pixsize, F, **kwargs):
    # Objective: To insert scalebar in a micrograph.
    # Note: this function is usually called by MyImg.scalebar method.
    # => see the original myimg.api.MyImg.scalebar method for more info.
    #-----
    
    # (0) Print messages if requested
    messages = kwargs.get('messages') or False
    if messages:
        print('Insert scalebar, start of function:')
        print(f'  Image  : {my_img.name}')
        print(f'  Pixels : {my_img.width}x{my_img.height}')
    
    # (1) Determine pixelsize = analyze obligatory pixsize argument.
    pixel_size = get_pixel_size(my_img, pixsize)
    # Print info if requested
    if messages:
        print('Determined pixel size:')
        print(f'  Pixsize argument : {pixsize}')
        print(f'  Final pixel size : {pixel_size}')

    # (2) Read multiplicative factor F
    # (If F is not given, set it to 1 and ...
    # (  in the following code multiply all relevant dimensions by F.
    # (* We CANNOT multiply directly myimg.settings.Scalebar variables
    # (  Reason is a bit tricky: In scripts processing multiple images ...
    # (  ... the 1st image scalebar dimensions would be multiplied by F
    # (  ... the 2nd image scalebar dimensions would be multiplied again => F*F
    if F is None: F = 1
    
    # (3) Initialize Scalebar = ScaleWithUnits object
    # (2a) Get lenght of scalebar (number + units)
    # (length is given as argument OR we calculate from img -size and pix-size
    length_arg = kwargs.get('length')
    if length_arg != None:
        length_of_scalebar = NumberWithUnits(length_arg)
    else:
        length_of_scalebar = calculate_scalebar_length(my_img, pixel_size, F)
    # (2b) Get length of scalebar in pixels
    length_in_pixels = \
        calculate_scalebar_length_in_pixels(length_of_scalebar, pixel_size)
    # (2c) Initialize scalebar = ScaleWithUnits object based on (2a, 2b, 2c)
    swu = ScaleWithUnits(
        number = length_of_scalebar.number,
        units  = length_of_scalebar.units,
        pixels = length_in_pixels)
    # (2d) Adjust scalebar lenght
    if length_arg != None:
        # If lenght argument was given, just round the lenght-in-pixels.
        swu.pixels = round(swu.pixels)
    else:
        # If length was calculated, adjust the size to a reasonable value.
        swu.adjust_scalebar_size()
    # (2e) Print info if requested
    if messages:
        print('Calculated scalebar size:')
        print(f'  {swu}')
    
    # (3) Collect additional keyword arguments.
    # (3a) key arguments are collected here
    position = kwargs.get('position') or Settings.Scalebar.position
    color    = kwargs.get('color')    or Settings.Scalebar.color
    bcolor   = kwargs.get('bcolor')   or Settings.Scalebar.bcolor
    # (3b) Special case - tranparent background: bcolor='transparent' => None
    # (bcolor=None would change bcolor to default = Settings.Label.bcolor
    if bcolor == 'transparent': bcolor = None
    # (3c) Additional arguments/parameters are collected in Settings.Scalebar
    # (they can be modified in code: Settings.Scalebar.font = 'arial.ttf'
    # (some of them are are read below for the sake of convenience    
    
    # (4) Calculate font size + prepare font
    font_name = Settings.Scalebar.font
    text_height_rel = Settings.Scalebar.text_height * F
    text_height_pix = round(text_height_rel * my_img.height) 
    font_size = my_img.get_font_size(font_name, text_height_pix)
    font_object = ImageFont.truetype(font_name, font_size)
    ImageDraw.fontmode = 'L'
    draw = ImageDraw.Draw(my_img.img)
    bbox = draw.textbbox((20, 20), swu.text(), font=font_object)
    scalebar_text_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
    if messages:
        print('Collected & calculated font parameters:')
        print(f'  Font name             : {font_name}')
        print(f'  Multiplicative factor : {F}')
        print(f'  Text_height_rel       : {text_height_rel}')
        print(f'  Text_height_pix       : {text_height_pix}')
        print(f'  Font_size             : {font_size}')
        print(f'  Scalebar text size    : {scalebar_text_size}')
            
    # (5) Calculate position and shape of the scalebar
    # (scalebar consists of: text and bar/line and box/background
    # (5a) Dimensions of text, bar and box 
    # ... xy-size of scalebar text (already calculated above)
    text_x, text_y = scalebar_text_size
    # ... xy-size of bar = scalebar line
    bar_x = swu.pixels
    bar_y = my_img.height * Settings.Scalebar.line_height * F
    # ... xy-size of box = scalebar background
    box_x = max(text_x,bar_x) + \
        my_img.height * (2 * Settings.Scalebar.edge_x * F) 
    box_y = text_y + my_img.height * (
        2 * Settings.Scalebar.edge_y * F +
        Settings.Scalebar.line_height * F + 
        Settings.Scalebar.separator * F)
    # (5b) (x1,y1) = position of the upper left corner of the box
    if position != None:
        x1 = position[0]
        y1 = position[1]
    else:
        # x1,y1 = position of the upper left corner of the box
        x1 = my_img.height * Settings.Scalebar.offset_x * F + box_x
        y1 = my_img.height * Settings.Scalebar.offset_y * F + box_y
        x1 = round(my_img.width  - x1)
        y1 = round(my_img.height - y1)
    # (5c) (x2,y2) position of the upper left corner of the bar/line
    x2 = round(x1 + my_img.height * Settings.Scalebar.edge_x * F)
    y2 = round(y1 + box_y - my_img.height * (
        Settings.Scalebar.edge_y * F +
        Settings.Scalebar.line_height * F)) 
    if text_x > bar_x:
        x2 = x2 + round((text_x - bar_x)/2)
    # (5d) (x3,y3) position of the text
    # (XY-anchor of the text will be 'mb' = middle,bottom
    # (x3 => anchor 'm' => in the middle of the box
    # (y3 => anchor 'b' => y-position of the bar/line - separator
    x3 = round(np.mean([x1, x1 + box_x]))
    y3 = round(y2 - my_img.height * Settings.Scalebar.separator * F)
    # (5e) Print info if requested
    if messages:
        print('Calculated scalebar geometry:')
        print('  Textsize in pixels       :', text_x, text_y)
        print('  Box, upper left corner   :', x1, y1)
        print('  Line, upper left corner  :', x2, y2)
        print('  Text, center (anchor=mb) :', x3, y3)
    
    # (6) Draw scalebar in image
    # Draw box/background 
    if bcolor != None: draw.rectangle([x1,y1,x1+box_x,y1+box_y], fill=bcolor)
    # Draw line
    draw.rectangle([x2,y2,x2+bar_x,y2+bar_y], fill=color)
    # Draw text
    draw.text((x3,y3),swu.text(), font=font_object, fill=color, anchor='mb')
    # Print final info if requested
    if messages:
        print('End of function, the scalebar drawn in the image object.')
    
    # (7) End of function.
    # (No return value; the scalebar is drawn in my_img.img object


def get_pixel_size(my_img, pixsize):
    # Objective: to get pixel size from pixsize argument + my_img object props.
    #-----
    
    # (1) Split pixsize argument to components
    pixsize_list = pixsize.split(',')
    pixsize_from = pixsize_list[0]
    pixsize_args = pixsize_list[1:]
    # (2) Decide how to determine pixel size value + call correct funtion 
    if   pixsize_from == 'rwi':
        pixel_size = pixel_size_from_rwi(my_img, pixsize_args)
    elif pixsize_from == 'knl':
        pixel_size = pixel_size_from_knl(my_img, pixsize_args)
    elif pixsize_from == 'mag':
        pixel_size = pixel_size_from_mag(my_img, pixsize_args)
    elif pixsize_from == 'txt':
        pixel_size = pixel_size_from_txt(my_img, pixsize_args)
    else:
        print('Unkwnown pixsize argument:', pixsize)
    # (3) Return final pixel size
    # (pixel_size type => myimg.utils.scalebar.NumberWithUnits
    return(pixel_size)


def pixel_size_from_rwi(my_img, pixsize_args):
    # Objective: to get pixel size from RWI + image_width.
    #-----
    
    # (1) RWI = the 1st pixsize argument after rwi (example: 'rwi,100um')
    rwi   = NumberWithUnits(pixsize_args[0])
    # (2) Pixel size = rwi / image_width_in_pixels (units = rwi.units)
    pixel = NumberWithUnits(
        number = rwi.number/my_img.width,
        units  = rwi.units)
    # (3) Return pixel_size (as pixel variable with type: NumberWithUnits)
    return(pixel)


def pixel_size_from_knl(my_img, pixsize_args):
    # Objective: to get pixel size from KNL.
    # (my_img is just a formal argument here
    # (my_img may be used in the future for beter error messages
    
    # (1a) KNL = the 1st pixsize agument after knl (example: 'knl,100um,202')
    knl = NumberWithUnits(pixsize_args[0])
    # (1b) Length in pixels = the 2nd pixsize arg  (example: 'knl,100um,202')
    try:
        length_in_pixels = float(pixsize_args[1])
    except Exception as err:
        print('Error in converting of lenght-in-pixels to float!')
        print(f'lenght-in-pixels: {length_in_pixels}')
        print(err)
        sys.exit()
    # (2) Pixel size = knl / length_in_pixels (units = knl.units)
    pixel = NumberWithUnits(
        number = knl.number/length_in_pixels,
        units  = knl.units)
    # (3) Return pixel_size (as pixel variable with type: NumberWithUnits)
    return(pixel)


def pixel_size_from_mag(my_img, pixsize_args):
    # Objective: to get pixel size from pixsize='mag...' argument.
    # Note: this works only for the calibrated microscopes
    # (calibrated microscope = a dataclass in myimg.settings.MicCalibrations
    #-----
    
    # (1) The first argument should be calibrated microscope name
    try:
        calibrated_microscope = getattr(
            Settings.MicCalibrations, pixsize_args[0])
    except AttributeError:
        print('Uknown microscope in pixsize argument!')
        print('Pixsize argument:', pixsize_args)
        sys.exit()
    
    # (2) Get magnification...
    # (2a) If the 2nd argument was given, it should be magnification
    if len(pixsize_args) == 2:
        mag = pixsize_args[1]
    # (2b) If 2nd argument was not given, get it from image name
    else:
        mag = re.search(r'\S+_(\d+\.?\d*[Kk]?[Xx]?)\.\S{3,}', my_img.name)[1]
        if mag == None:
            print('Unrecognized magnification in filename!')
            print('Filename:', my_img.name)
            sys.exit()
    # (2c) Convert the magnification to number
    # Replace final 'x' with the empty string
    if mag.endswith('x'):
        mag = mag.replace('x','')
    # Replace final 'k' with the three zeros (if it is there)
    if mag.endswith('k'):
        mag = mag.replace('k','000')
    # Convert magnification to float
    try:
        mag = float(mag)
    except ValueError:
        print('Unclear magnification in pixsize argument!')
        print('Pixsize argument:', pixsize_args)
        sys.exit()
        
    # (3) Now we should have calibrated_microscope name + magnification
    # Get calibration constant with units
    const = calibrated_microscope.const
    units = calibrated_microscope.units
    # Calculate RWI = real-width-of-image (simple relation: rwi = const/mag) 
    rwi = NumberWithUnits(
        number = const/mag,
        units  = units)
    # Set correct units for RWI
    # (this is not strictly necessary, but it is more correct
    rwi.set_correct_units()
    # Recalculate RWI to pixel_size
    pixel = NumberWithUnits(
        number = rwi.number/my_img.width,
        units  = rwi.units)
    
    # (4) Return pixel_size (as pixel variable with type: NumberWithUnits)
    return(pixel)
    

def pixel_size_from_txt(my_img, pixsize_args):
    # Objective: to get pixel size from  pixsize='txt...' argument.
    # Note: this works only for microscopes with known description files.
    # (known descr.file = a dataclass in myimg.settings.MicDescriptionFiles
    #-----
    
    # (1) Get the name of mic_description_file class from pixsize_args
    # Example:
    #  - we have a string: pixsize_args='MAIA' 
    #  - and we need an object name: Settings.MicDescriptionFiles.MAIA
    # => GoogleSearch: python access variable/object/class when I know string
    # => https://stackoverflow.com/q/9437726
    # => https://stackoverflow.com/q/1167398
    microscope = pixsize_args[0]
    mic_description_file = getattr(
        Settings.MicDescriptionFiles, microscope)

    # (2) Get the text description file name 
    # once we know the mic_description_file class from the previous step
    # Example:
    #  - we have: my_img object + mic_description_class object
    #  - and we need text description file name, such as: 'FIGS\img123.hdr'
    img_filename_without_extension = os.path.splitext(my_img.name)[0]
    text_description_filename_general = mic_description_file.filename
    text_description_filename = \
        text_description_filename_general.replace(
            '*', img_filename_without_extension)
    # Yet final conversion - some description files can contain wildcards
    # Example: Settings.MicDescriptionFiles.MAIA.filename = '*-???.hdr'
    # Trick to solve this: convert wildcards to filename using glob function
    text_description_filename = glob.glob(text_description_filename)[0]
    
    # (3) Open the text description file and get pixel size
    # (a) Prepare varialbles
    pixsize_regexp = re.compile(mic_description_file.pixsize_line)
    pixsize_units  = mic_description_file.pixsize_units
    pixsize_number = None
    # (b) Search for pixel size in the text description file
    # (considering all possible exceptions, as specified below
    try:
        fh = open(text_description_filename)
        for line in fh:
            m = pixsize_regexp.search(line)
            if m:
                pixsize_number = float(m.group(1))
                break
    # (c) Considering all possible exceptions ...
    except FileNotFoundError:
        print('Error when trying to read image description file!')
        print(f'File [{text_description_filename}] not found.')
        sys.exit()
    except Exception as err:
        print('Error when trying to read image description file!')
        print('Exception name:', type(err).__name__)
        print('Description:', err)
        sys.exit()

    # (4) Create NumberWithUnits object
    # (If pixel size was not found => pixsize_number = None
    # (exception will be raised automatically - do do not have to worry.
    # (The exception is treated directly in NumberWithUnits class.
    pixel = NumberWithUnits(
        number = pixsize_number,
        units  = pixsize_units)

    # (5) Return pixel_size (as pixel variable with type: NumberWithUnits)
    return(pixel)


def calculate_scalebar_length(my_img, pixel_size, F):
    # Objective: to calculate suitable scalebar length,
    # based on image size, pixel size, Settings.Scalebar.length and factor F.
    # (Note: we calculate just length+units;
    # ( lenght-in-pixels is calcualted in the next step
    # (Reason: step-by-step approach - we will do this in the next function;
    # ( for length-in-pixels we need the same units of scalebar and pixel_size
    #-----
    
    # (1) Determine RWI from image size and pixel size.
    rwi = NumberWithUnits(
        number = my_img.width * pixel_size.number,
        units  = pixel_size.units)
    
    # (2) Determine suitable lenght of scalebar
    # (At the moment, we determine just number+units
    # (  Lenght-in-pixels will be determined later
    # (  Reason: for lenght-in-pixels we have to consider pixel-size units
    nwu = NumberWithUnits(
        # Scalebar length should be a fraction of image width
        # This fraction of image width is saved in Settings.Scalebar.lenght
        # Moreover, if multiplicative coeff F was given, is should be Fx more.
        number = rwi.number * Settings.Scalebar.length * F,
        units  = rwi.units)
    return(nwu)
    
    
def calculate_scalebar_length_in_pixels(length_of_scalebar, pixel_size):
    # Objective: to calculate scalebar length-in-pixels
    # (using two NumberWithUnits objects: length_of_scalebar + pixel_size
    #-----   
    # (1) pixel_size.units should be the same as length_of_scalebar.units
    pixel_size.set_units_to(length_of_scalebar.units)
    # (2) If units are the same, the calculation is easy
    # Example: scalebar = 5um, pixel_size = 0.1um => length_in_pix = 5/0.1 = 50 pix
    length_in_pixels = length_of_scalebar.number / pixel_size.number
    # (3) Return calculated length_of_scalebar_in_pixels
    return(length_in_pixels)
    
'''
Module: myimg.settings
----------------------
Data module with default settings and calibrations for package myimg.

The module contains several dataclasess:

* myimg.settings.Scalebar
  = default parameters for drawing of scalebars
* myimg.settings.Label
  = default parameters for drawing of image labels
* myimg.settings.MicCalibrations
  = class contaning the microscope calibrations
* myimg.settings.MicDescriptionFiles
  = class describing the microscope description files
'''


from dataclasses import dataclass
from typing import Any


@dataclass
class Scalebar:
    '''
    Default parameters of scalebars (dimensions, font, ...).
    
    * most dimensions in this section = fractions of image height
    * only default/precalculated lenght of scalebar = 1/6 * image_width
    * default/precalculated position = lower-right corner (if position=None)
    '''
    length      : float = 0.167   # default lenght of scalebar
    font        : str = 'timesbd.ttf'  # text font
    text_height : float = 0.040   # text height
    line_height : float = 0.010   # thickness of scalebar line
    line_border : float = 0.0015  # thickness of scalebar line border
    separator   : float = 0.010   # space between scalebar line and text
    edge_x      : float = 0.04    # left and right edge
    edge_y      : float = 0.02    # upper and lower edge
    offset_x    : float = 0.02    # horizontal distance from the edge of image
    offset_y    : float = 0.02    # vertical   distance from the edge of image
    position    : Any = None      # scalebar position: None or (X,Y) coords
    color       : Any = 'black'   # color of text and line
    bcolor      : Any = 'white'   # color of backgound


@dataclass
class Label:
    '''
    Default parameters of image labels (dimensions, font, ...).
    
    * all dimensions in this section = multiples of image height
    * the default font size is similar as in the case of Scalebar
    '''
    font        : str   = 'calibri.ttf'  # text font
    text_height : float = 0.044    # text height
    text_offset : float = 0.025    # text offset to adjust XY text position
    color       : Any   = 'black'  # color of text
    bcolor      : Any   = 'white'  # color of backgound
    
    
@dataclass
class Caption:
    '''
    Default parameters of image Captions (dimensions, font, ...).
    
    * all dimensions in this section = multiples of image height
    * the default font size is similar as in the case of Scalebar
    '''
    font        : str   = 'calibri.ttf'  # text font
    text_height : float = 0.036    # text height
    text_offset : float = 0.025    # text offset to adjust XY text position
    color       : Any   = 'black'  # color of text
    bcolor      : Any   = 'white'  # color of backgound
    align       : str   = 'center' # text horizontal alignment
    

@dataclass
class MicCalibrations: 
    '''
    Microscope calibration constants.
    
    * This dataclass is a container for several sub-dataclasses.
    * The subdataclasses define the individual calibrated microscopes.
    
    Calculation of calibration constant for given microscope
    
    * mag   = magnification, for which we know rwi
    * rwi   = real width of image (in our case given in [mm]
    * const = calibration constant = rwi * mag
    
    Usage of calibration constants
    
    * rwi at given mag: rwi = const/mag (const = calibration constant)
    * Warning: for EM microscopes, these relations may be just approximate
    
    Justification
    
    * 2x higher mag => 2x lower rwi
    * 4x higher mag => 4x lower rwi
    * Therefore (according to elementary logic)
        - rwi = some_constant/mag
        - const = some_constant = calibration_constant = rwi * mag
        - physical meaning of calibration const:
          for given microscope, const = rwi of image at mag=1x
          
    Usage of this dataclass => subclasses of this dataclass
        
    * This dataclass is usually not used directly.
    * The data are employed within the myimg package.
    * Nevertheless, a short example follows:
        
        >>> import myimg.settings as Settings
        >>> print(Settings.MicCalibrations.TecnaiVeleta.const)
    '''
    

    @dataclass
    class LM_Nikon1:
        '''
        Calibration of Nikon microscope with ProgRes camera.
        
        * Typical image size = [1024x768]pix
        * Alternative image sizes = integer multiples possible due to binning.
        * Binning does not influence real-width-of-image and calibration const.
        '''
        description : str = 'LM Nikon, ProgRes camera'
        const       : float = 6.55300   # const = rwi * mag (rwi = const/mag)
        units       : str = 'mm'        # const units


    @dataclass
    class LM_Nikon2:
        '''
        Calibration of Nikon microscope with Basler camera.
        
        * Typical image sizes = [2464x2056]pix, [1232x1028]pix ...
        * Alternative image sizes = integer multiples possible due to binning.
        * Binning does not influence real-width-of-image and calibration const.
 
        '''
        description : str = 'LM Nikon, Basler camera'
        const       : float = 8.50000   # const = rwi * mag (rwi = const/mag)
        units       : str = 'mm'        # const units


    @dataclass
    class TecnaiVeleta:
        '''
        Calibration of Tecnai microscope with Veleta camera.
        
        * Typical image size = [1024x1024]pix
        * Alternative image sizes = integer multiples possible due to binning.
        * Binning does not influence real-width-of-image and calibration const.
        '''
        description : str = 'TEM Tecnai, Veleta3G camera'
        const       : float = 110.440  # const = rwi * mag (rwi = const/mag)
        units       : str = 'mm'       # const units
   

     
@dataclass
class MicDescriptionFiles:
    '''
    Microscope description files.
    
    * Some microscopes yield micrographs *with* text description files.
    * The description file contains additional info about the micrograph.
        - the filename is usually similar to the micrograph/image file
        - the description file contains information about the pixel size
        - therefore, a description file can be used
          for the micrograph calibration
    '''
    
    # MicroscopeDescrFiles = microscopes with description files
    # (description file is a text file containing pixelsize
    # (for some microscopes, decription files are saved together with images
    # (each microscope below = dictionary containing the following elements
    # (...filename = name of descr.file; * = name of img.file, ? = any char
    # (...pixsize_line = regular expression describing line with pixelsize
    # (   the paretheses () within the regexp should catch the pixelsize number
    # (...pixsize_units = units, in which the pixelsize is given

    
    @dataclass
    class MAIA:
        '''
        Desription files produced by an SEM microscope MAIA.
        '''
        filename     : str = r'*-???.hdr'
        pixsize_line : str = r'^PixelSizeX=(.*)'
        pixsize_units: str = r'm'

    
    @dataclass
    class VEGA:
        '''
        Description files produced by an SEM microscope VEGA.
        '''
        filename     : str = r'*.bhd'
        pixsize_line : str = r'^Pix=(.*)'
        pixsize_units: str = r'um'


    @dataclass
    class JEOL:
        '''
        Description files produced by an SEM microscope JEOL.
        '''
        filename     : str = r'*.hdr'
        pixsize_line : str = r'^Pix: (\S+)'
        pixsize_units: str = r'um'

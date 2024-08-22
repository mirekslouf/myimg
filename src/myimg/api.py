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

More examples are spread all over the documentation.
    
1. How to use myimg.objects:
    - myimg.objects.MyImage = single image = the basic object with many methods
    - myimg.objects.Montage = multi-image = a rectangular grid of images
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


class MyImage(myimg.objects.MyImage):
    '''
    Class defining MyImage objects.
    
    * MyImage object = image-name + PIL-image-object + various methods.
    * This class is just inherited from myimg.objects.MyImage.
    * More help: https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html 
    '''
    pass

class Montage(myimg.objects.Montage):
    '''
    Class defining Montage objects.
    
    * Montage object = a rectangular multi-image.
    * This class is just inherited from myimg.objects.Montage. 
    * More help: https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html 
    '''
    pass


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

    def fourier(img):
        pass


class Settings:
    '''
    Settings for package myimg.
    
    * This class imports all classes from myimg.settings.
    * Thanks to this import, we can use Settings myimg.api as follows:
        
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

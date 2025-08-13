'''
Module: myimg.utils.label
-------------------------

This module defines function insert_label, with the following features:

* The function inserts a label to the upper left corner of an image.
* The function is in this separated module as it is a bit longer.
* The function is usually not called directly, but through myimg.api:
    
>>> # Inserting scalebar using myimg.api interface
>>> import myimage.api as mi
>>>
>>> img = mi.MyImage('somefile.bmp')
>>>
>>> # This calls myimg.utils.label.insert_label function
>>> img.label('a')
>>>
>>> img.save_with_ext('_l.png') 

Notes to documentation:

* The function *insert_label* in this module does not have a docstring,
  as it is not called directly. 
* The docstring with detailed description of all parameters can be found
  in the calling myimg.api.MyImage.label method.
  (the usage of the label method is shown in the example above).
'''


import myimg.settings as Settings
from PIL import ImageFont, ImageDraw


def insert_label(my_img, label, F, **kwargs):
    # Objective: To insert label in the upper left corner of an image.
    # Note: this function is usually called by MyImg.label method.
    # => see the original myimg.api.MyImg.scalebar method for more info.
    #-----
    
    # (1) Read multiplicative factor F
    # (If F is not given, set it to 1 and ...
    # (  in the following code multiply all relevant dimensions by F.
    # (* We CANNOT multiply directly myimg.settings.Scalebar variables
    # (  Reason is a bit tricky: In scripts processing multiple images ...
    # (  ... the 1st image scalebar dimensions would be multiplied by F
    # (  ... the 2nd image scalebar dimensions would be multiplied again => F*F
    if F is None: F = 1
    
    # (2) Collect optional keyword arguments.
    color  = kwargs.get('color')  or Settings.Label.color
    bcolor = kwargs.get('bcolor') or Settings.Label.bcolor
    # Special case: tranparent background
    # (bcolor=None would change bcolor to default = Settings.Label.bcolor
    # (bcolor='transparent' is now INTENTIONALLY changed to None => no bkg
    if bcolor == 'transparent': bcolor = None
    # Other optional parameters are read from Settings.Label.
    # (some of them read here for the sake of convenience
    # (all can be changed in code, such as: Settings.Label.text_height=1/10
    font            = Settings.Label.font
    text_height_rel = Settings.Label.text_height * F
    text_height_pix = text_height_rel * my_img.height
    
    # (3) Determine font size
    fontsize = my_img.set_font_size(font, text_height_pix)
    offset   = my_img.width * Settings.Label.text_offset * F
    
    # (4) Create label
    # ...initialize drawing
    draw = ImageDraw.Draw(my_img.img)
    # ...initialize font
    my_font = ImageFont.truetype(font, fontsize)
    # ...get height and width of current label
    # my_label_height = fontsize
    my_label_height = text_height_pix
    my_label_width  = my_font.getlength(label)
    # ...draw background box if required
    # (constants around offset - empirical, to get a reasonable backround
    if bcolor is not None:
        draw.rectangle(
            [0, 0, my_label_width + 2*offset, my_label_height + 1.45*offset], 
            fill=bcolor)
    # ...draw text inside the background box
    # (constants around offset = empirical, to get good centering of font
    draw.text((offset,offset//3), label, font=my_font, fill=color)   
    
    # (5) Done. The label is drawn/saved in self.img

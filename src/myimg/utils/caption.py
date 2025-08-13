'''
Module: myimg.utils.caption
---------------------------

This module defines function insert_caption, with the following features:

* The function adds a single-line caption at the bottom of an image.
* The function is in this separated module as it is a bit longer.
* The function is usually not called directly, but through myimg.api:
    
>>> # Inserting caption using myimg.api interface
>>> import myimage.api as mi
>>>
>>> img = mi.MyImage('somefile.bmp')
>>>
>>> # This calls myimg.utils.caption.insert_caption
>>> img.caption('This is my image.')
>>>
>>> img.save_with_ext('_t.png') 

Notes to documentation:

* The function *insert_caption* in this module does not have a docstring,
  as it is not called directly. 
* The docstring with detailed description of all parameters can be found
  in the calling myimg.api.MyImage.caption method.
  (the usage of the caption method is shown in the example above).
'''


import myimg.settings as Settings
from PIL import ImageFont, ImageDraw


def insert_caption(my_img, text, F, **kwargs):
    # Objective: To insert caption at the bottom of an image.
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
    color  = kwargs.get('color')  or Settings.Caption.color
    bcolor = kwargs.get('bcolor') or Settings.Caption.bcolor
    align  = kwargs.get('align')  or Settings.Caption.align
    # Special case: tranparent background
    # (bcolor=None would change bcolor to default = Settings.Label.bcolor
    # (bcolor='transparent' is now INTENTIONALLY changed to None => no bkg
    if bcolor == 'transparent': bcolor = None
    # Other optional parameters are read from Settings.Label.
    # (some of them read here for the sake of convenience
    # (all can be changed in code, such as: Settings.Label.text_height=1/10
    font            = Settings.Caption.font
    text_height_rel = Settings.Caption.text_height * F
    text_height_pix = text_height_rel * my_img.height
    
    # (3) Determine font size
    fontsize = my_img.set_font_size(font, text_height_pix)
    offset   = my_img.width * Settings.Caption.text_offset * F
    
    # (4) Create label
    # Initialize drawing
    draw = ImageDraw.Draw(my_img.img)
    # Initialize font
    my_font = ImageFont.truetype(font, fontsize)
    # Get height and width of current label
    caption_text_height = text_height_pix
    caption_bar_height  = int(round(caption_text_height + 1.50*offset))
    # Add bottom bar at the bottom of an image
    my_img.border(border=(0,0,0,caption_bar_height), color=bcolor)
    # Draw the text into the bottom bar
    # (i) Re-initialize drawing
    # (reason: border employs PIL.ImageOps.expant => creates new image object!
    draw = ImageDraw.Draw(my_img.img)
    # (ii) prepare parameters for drawing: x_pos, x_anchor
    # (in draw.text, the single-line text alignment => anchor argument
    if type(align) == int:
        x_pos = align
        x_anchor = 'la'
    elif align in ('left','Left'):
        my_offset = (offset // 4) if align == 'Left' else offset
        x_pos = my_offset
        x_anchor = 'la'
    elif align in ('right','Right'):
        my_offset = (offset // 4) if align == 'Right' else offset
        x_pos = my_img.width - my_offset
        x_anchor = 'ra'
    elif align == 'center':
        x_pos = my_img.width // 2
        x_anchor = 'ma'
    else:
        print('Image, figure caption - uknown alignment - setting default!')
        x_pos = my_img.width // 2
        x_anchor = 'ma'
    # (iii) prepare the last parameter for drawing: y_pos
    y_pos = my_img.height - caption_bar_height + offset//3
    # (iv) draw the text
    draw.text((x_pos, y_pos),
        text, font=my_font, fill=color, anchor=x_anchor)
    
    # (5) Done. The label is drawn/saved in self.img

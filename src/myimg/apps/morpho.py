'''
Module: myimg.apps.morpho
-------------------------

Morphology analysis of binary micrographs. 

Limitations:

* Under development.
* Functions are added gradually.
'''

import numpy as np
from PIL import Image

import myimg.api
from pathlib import Path

def read_binary_image(binary_image):
    '''
    Read image or convert an array to a boolean binary image.
    '''
    
    # Step 1 :: Read image
    
    # Image = MyImage object => read the image from the object.
    if isinstance(binary_image, myimg.api.MyImage):
        arr = np.asarray(binary_image.img)
    # Image = str => path => open image, convert to grayscale, save as array.
    elif isinstance(binary_image, (str, Path)):
        arr = np.asarray(Image.open(binary_image).convert("L"))
    # Image = something else that can be converted to image (PIL object).
    else:
        arr = np.asarray(binary_image)

    # Step 2 :: Convert image to binary
    
    # Convert to binary => any non-zero value is treated as white.
    # Trick: this returns boolean array with False/True ~ 0/1 values. 
    return (arr > 0)


def MLI(binary_image, angles=(0,90)):
    """
    Calculate Mean Linear Intercept (MLI) of a binary image.

    Parameters
    ----------
    binary_image : str or numpy.ndarray
        Filename of a binary image, or a NumPy array.
        Zero = black phase, non-zero = white phase.

    angles : int or sequence of int
        Direction(s) of the test lines in degrees.
        Supported values: 0, 45, 90, 135.

    Returns
    -------
    float
        MLI in pixels.

        If several angles are supplied, the returned value
        is the arithmetic mean of the MLIs obtained in all
        requested directions.

    Notes
    -----
    MLI is calculated from the lengths of all continuous
    black and white segments ("chords") intercepted by the
    test lines.

    For 45 and 135 degree directions, the pixel length of
    one diagonal step is sqrt(2), so the MLI is corrected
    accordingly.
    """

    # Local functions ---------------------------------------------------------
    
    def intercepts(line):
        """
        Return lengths of all continuous runs in a 1-D binary line.
        
        * function created by AI
        * then commented and slightly modified
        """
        
        # (1) Get indices of of elements
        #     where black-changes-to-white or white-changes-to-black.
        # (np.flatnonzero => return indices of non-zero elements
        # (line[1:] = from 1st to the end, line[:-1] from 0th to last but one
        # (line[1:] != line[:-1] => True if 1st/0th, 2nd/1st, ... differ
        # (add 1 at the end to correct for real index => change not in 0th ...
        changes = np.flatnonzero(line[1:] != line[:-1]) + 1
        
        # (2) Get limits = [0, changes, len(line)]
        #     i.e. arr-of-limits = arr-of-changes with first and last element.
        # (np.r_ = trick = fast concatenation of arrays
        # (np.r_[0, [1,2,3], 10]) = [0,1,2,3,10]
        limits = np.r_[0, changes, len(line)]

        # (3) Calculate all intercepts
        all_intercepts = [
            limits[i + 1] - limits[i]
            for i in range(len(limits) - 1)
        ]
        
        # (4) Return the result
        return(all_intercepts)


    def lines(image, angle):
        """
        Generate 1-D lines in the requested direction.
        """
        if angle == 0:
            # Horizontal lines
            return list(image)
        elif angle == 90:
            # Vertical lines
            return [image[:, i] for i in range(image.shape[1])]
        elif angle == 45:
            # Diagonal lines (45deg)
            flipped = np.fliplr(image)
            return [
                np.diagonal(flipped, offset=k)
                for k in range(-image.shape[0] + 1, image.shape[1])
            ]
        elif angle == 135:
            # Diagonal lines (135deg)
            return [
                np.diagonal(image, offset=k)
                for k in range(-image.shape[0] + 1, image.shape[1])
            ]
        else:
            # Other value => error => raise exception.
            # (here: just for completness
            # (the {angle} values already pre-checked in the main function
            raise ValueError("Supported angles: 0, 45, 90 and 135 deg.")

    # Main function -----------------------------------------------------------
    
    # (1) Read image using {binary_image} argument
    # (universal func: reads MyImage-objects, PIL-objects, arrays, finenames
    image = read_binary_image(binary_image)

    # (2) Check and adjust {angles} argument
    # (convert {angles} to list
    # (it might be scalar = single angle, or something more exotic 
    if np.isscalar(angles): angles = [angles]
    else: angles = list(angles)
    # (check if {angles} are allowed values
    for angle in angles:
        if angle not in (0, 45, 90, 135):
            raise ValueError( "Supported angles: 0, 45, 90 and 135 deg.")

    # (3) Calculate average MLI
    
    # (3a) Prepare empty list for MLI values
    # (we calculate MLI in for all {angles} = directions
    # (the calculations are  saved in MLIs and averaged at the end
    MLIs = []

    # (3b) Calculate MLI for each {angle} = direction
    # (all_intercepts_for_angle = list of ALL intercepts for given angle/dir
    # (MLI_for_angle = mean linear intercept for given angle/directions
    # (MLIs = list of all MLI_for_angle intercepts => averaged at the end
    for angle in angles:
        # Prepare empty list of all intercepts
        all_intercepts_for_angle = []
        # Calculate intercepts for given angle
        for line in lines(image, angle):
            all_intercepts_for_angle.extend(intercepts(line))
        # 
        if all_intercepts_for_angle:
            MLI_for_angle = np.mean(all_intercepts_for_angle)

            # Correct diagonal chord lengths
            if angle in (45, 135): MLI_for_angle = MLI_for_angle * np.sqrt(2)

            MLIs.append(MLI_for_angle)

    # (3c) Calculate and return final MLI
    # (average of all MLIs = all MLIs for all angles/directions
    MLI = float(np.mean(MLIs))
    return(MLI)

'''
Module: myimg.apps.profiles
---------------------------

Radial and azimuthal profiles for package myimg.
'''

# Import modules
# --------------
import sys
import numpy as np
import matplotlib.pyplot as plt
# Reading input images
from PIL import Image
import myimg.api


class RadialProfile:
    '''
    Class defining *RadialProfile* object.
    '''
                
    
    def __init__(self, img, center=None):
        
        # (1) Process the first argument = image
        # (the first argument = input array
        # (it can be: np.array, MyImage object, PIL object or path-to-image
        if type(img) == np.ndarray:
            # img comes as numpy array
            # (the simplest case - just assign img to arr
            arr = img
        elif type(img) == myimg.api.MyImage:
            # img comes as MyImage object
            # (check the image type and convert it to array if possible
            if img.itype in ('binary','gray','gray16'):
                arr = np.asarray(img.img)
            else:
                print('Radial profiles only for binary or grayscale images!')
                sys.exit()
        elif type(img) == str:
            # image comes as str => we assume it is an image name
            # TODO: check the image type
            img = Image.open(img)
            arr = np.asarray(img)
        else:
            print('Unknown image type!')
            sys.exit()
            
        # (2) Process the 2nd argument = center
        if center is None:
            x = arr.shape[0]/2 
            y = arr.shape[1]/2
        else:
            x = center[0]
            y = center[1]
            
        # (3) Calculate the radial profile
        R,I = RadialProfile.calc_radial(arr, center=(x,y))
        self.R = R
        self.I = I
    
    
    @staticmethod
    def calc_radial(arr, center):
        
        # (1) Get image dimensions
        (width,height) = arr.shape
        
        # (2) Unpack center coordinates
        xc = center[0]
        yc = center[1]
        
        # (3) Calculate radial distribution
        # --- (3a) Prepare 2D-array/meshgrid with calculated radial distances
        # (trick: the array/meshgrid will be employed for mask
        # (...the meshgrid size = the same as the original array size
        [X,Y] = np.meshgrid(np.arange(width)-xc, np.arange(height)-yc)
        R = np.sqrt(np.square(X) + np.square(Y))
        # --- (3b) Initialize variables
        radial_distance = np.arange(1,np.max(R),1)
        intensity       = np.zeros(len(radial_distance))
        index           = 0
        bin_size        = 1
        # --- (3c) Calcualte radial profile
        # (Gradual calculation of average intenzity
        # (in circles with increasing distance from the center 
        # (trick 2: to create the circles, we will employ mask from trick 1
        for i in radial_distance:
            mask = np.greater(R, i - bin_size) & np.less(R, i + bin_size)
            values = arr[mask]
            intensity[index] = np.mean(values)
            index += 1 
        
        # (4) Save profile to array, save it to file if requested, and return it
        profile = np.array([radial_distance, intensity])
        return(profile)
    

    def show(self):
        # TODO :: nicer plot
        plt.plot(self.R, self.I)
        plt.show()
    

    def save(self):
        # TODO :: save radial distribution as file (with scale, if available)
        pass
    
    

class AzimuthalProfile:
    '''
    Class defining *AzimuthalProfile* object.
    '''
    # TODO :: define in analogy with *RadialProfile* class
    # (this should be done after finishing the code and docs for RadialProfile
    pass

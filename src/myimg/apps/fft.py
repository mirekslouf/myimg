'''
Module: myimg.apps.fft
-----------------------

Fourier transform utilities for package myimg.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack



class FFT:
    '''
    TODO: add documentation ...
    '''


    def __init__(self, img):
        if img.itype != 'gray':
            print('Fourier transform works only for grayscale images!')
            print('End of program.')
            sys.exit()
        a = np.asarray(img.img)
        b = scipy.fftpack.fft2(a)
        b = scipy.fftpack.fftshift(b)
        self.fft = b
        self.intensity = np.abs(b)
        self.phase = np.angle(b)
            
        
    def show(self, 
             what='intensity', axes=False, cmap=None, icut=None):
        if what == 'intensity':
            a = self.intensity
        else:
            a = self.phase
        if cmap is None:
            cmap = 'gray'
        plt.imshow(a, cmap=cmap, vmax=icut)
        if axes == False: plt.axis('off')
        plt.show()
        
    
    def save(self, filename, dpi=300,
             what='intensity', axes=False, cmap=None, icut=None):
        #TODO
        pass
    
    
    def normalize(self, normalize='16bit'):
        # (1) Determine the normalization constant = final max.intensity.
        # (the values in self.intensity will be normalized to max_intensity
        if normalize == '16bit':
            norm_constant = 2**16 - 1
        elif normalize == '8bit':
            norm_constant = 2**8 - 1
        elif type(normalize) in (float,int):
            norm_constant = normalize
        else:
            print('Module myimg.apps.fft -> wrong normalize argument!')
            print('Possible values: "16bit", "8bit", or a number.')
            print('No normalization performed.')
        # (2) Perform the normalization
        max_intensity = np.max(self.intensity)
        self.intensity = np.int32( 
            np.round(self.intensity/max_intensity * norm_constant))            
    
    
    def radial(self):
        # TODO
        pass

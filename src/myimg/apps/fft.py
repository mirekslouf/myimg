'''
Module: myimg.apps.fft
-----------------------

Fourier transform utilities for package myimg.
'''

# Import modules
# --------------
import sys
# Plotting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1  # to add nice colorbars
# FFT
import scipy.fftpack
# Reading input images
from PIL import Image
import myimg.api


class FFT:
    '''
    Class defining FFT object.
    '''


    def __init__(self, img):
        '''
        Initialize FFT object.
        
        >>> # Simple usage of FFT objects
        >>> import myimg.api as mi        # standard import of myimg
        >>> img = mi.MyImage('some.png')  # open an image using myimg.api
        >>> fft = my.Apps.FFT(img)        # calculate FFT of the img object
        >>> fft.show(cmap='magma')        # show the result

        Parameters
        ----------
        img : image (array or str or path-like or MyImage object)
            The 2D object, from which we will calculate FFT
            = 2D-DFFT = 2D discrete fast Fourier transform.

        Returns
        -------
        FFT object.
        
        Technical details
        -----------------
        * FFT object, 3 basic attributes: FFT.fft (array of complex numbers),
          FFT.intensity (array of intensities = magnitudes = real numbers)
          and FFT.phase (array of phases = angles in range -pi:pi).
        * FFT object is pre-processed in the sense that the intensity center
          is shifted to the center of the array (using scipy.fftpack.fftshift).
        * FFT object carries the information about calibration (pixel-size),
          on condition it was created from MyImage object (the typical case).
        '''
        
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
                print('Fourier transform only for binary or grayscale images!')
                sys.exit()
        elif type(img) == str:
            # image comes as str => we assume it is an image name
            # TODO: check the image type
            img = Image.open(img)
            arr = np.asarray(img)
        else:
            print('Unknown image type!')
            sys.exit()
        
        # (2) Calculate FFT, shift the origin, and save the result
        arr = scipy.fftpack.fft2(arr)
        arr = scipy.fftpack.fftshift(arr)
        self.fft = arr
        self.intensity = np.abs(arr)
        self.phase = np.angle(arr)
        
        # (3) Save the information about pixel size, if available
        if type(img) == myimg.api.MyImage:
            # TODO: extend rec.units, add units here
            # TODO: consider adding set_scale, scalebar to fft package
            # self.recpixsize = 1 / (img.width * img.pixsize.number)
            pass
        else:
            self.recpixsize = None
            
    
    def normalize(self, what='intensity', itype='16bit', icut=None):
        '''
        Normalize results of fft calculation.
        
        Parameters
        ----------
        what : str, optional, default is 'intensity'
            What result should be normalized - 'intensity' or 'phase'.
            Intensity normalization = from arbitrary scale to given itype.
            Phase normalization = from (-pi:pi) to (0:2*pi) in order to
            eliminate negative values, which cause problems in plotting/saving.
        itype : str, optional, default is '16bit'
            Format of the normalized fft arrays (that can be saved as images).
            If '16bit' (default), the images will be 16-bit.
            If '8bit' (less suitable, narrow range) the images will be 8-bit.
        icut : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        '''
        
        # Define local functions

        def normalize_intensity(norm_constant):
            # Intensity is a non-negative number
            # and so it can be normalized in a standard way.
            
            # (1) Dermine max.value = maximal intenstity in the array.
            max_intensity = np.max(self.intensity)
            
            # (2) Standard normalization to maximal value
            # BUT taking into account rounding and type of the final array.
            # (a) Standard normalization
            self.intensity = self.intensity/max_intensity * norm_constant
            # (b) Adjusting type of the normalized array
            # If the normalization constant is an integer value,
            # ..round the result to int and convert the array to integers
            # ..this is important for the smooth converting of arrays to images
            if type(norm_constant) == int:
                self.intensity = np.round(self.intensity).astype('int')
        
        def normalize_phase(norm_constant):
            # Phase takes the values in interval (-pi;pi),
            # BUT for saving phase as image we need positive values
            # THEREFORE, if we want to normalize phase for plotting,
            # we have to convert phase to range (0;2*pi) and then normalize.
            
            # (1) Convert (-pi:pi) to (0:2*pi)
            # for the reason explained above
            self.phase = np.where( 
                self.phase < 0, self.phase + 2*np.pi, self.phase)
            
            # (2) Max.phase = upper limit should be ALWAYS 2*pi
            # even if this specific number is not in the array.
            max_phase = 2*np.pi
            
            # (3) Standard normalization to maximal value
            # BUT taking into account rounding and type of the final array.
            # (a) Standard normalization
            self.phase = self.phase/max_phase * norm_constant
            # (b) Adjusting type of the normalized array
            # If the normalization constant is an integer value,
            # ..round the result to int and convert the array to integers
            # ..this is important for the smooth converting of arrays to images
            if type(norm_constant) == 'int':
                self.phase = np.round(self.phase).astype('int')
        
        # Code of the method itself (after defning local functions)
        
        # (1) Calculate normalization constant        
        if itype == '16bit': norm_constant = 2**16 - 1
        elif itype == '8bit': norm_constant = 2**8 - 1 
        else: norm_constant = itype
        
        # (2) Determine, what to normalize and perform the normalization(s).
        if what == 'intensity':
            normalize_intensity(norm_constant)
        elif what == 'phase':
            normalize_phase(norm_constant)
        else:  # Exit if what argument was wrong. 
            print('myimg.apps.fft.convert_to_16big -> wrong what argument!')
            sys.exit()
            
        # (3) Perform intensity cut if required (and re-normalize)
        if (what == 'intensity') and (icut is not None):
            arr = self.intensity
            self.intensity = np.where(arr > icut, icut, arr)
            normalize_intensity(norm_constant)
    
        
    def show(self, what='intensity',
             axes=False, cmap=None, icut=None, colorbar=False, 
             output=None, dpi=300):
        '''
        Show FFT object = Fourier transform of an image.

        Parameters
        ----------
        what : str, optional, default is 'intensity'
            What result should be shown - 'intensity' or 'phase'.
        axes : bool, optional, default is False
            Show axes around the plotted/shown image.
        cmap : str, optional, default is None
            Matplotlib cmap name, such as 'magma' or 'viridis'.
        icut : int or float, optional, default is None
            Intensity cut value.
            If icut = 1000, all intenstity values >1000 are set to 1000.
        colorbar : bool, optional, the default is False
            If True, a colorbar is added to the plot.
        output : str or path-like object, optional, default is None
            If output argument is given, the plot is saved to {output} file.
        dpi : int, optional, default is 300
            DPI of the saved image.
            Relevant only if ouput is not None.

        Returns
        -------
        None
            The output is the image/plot of the Fourier transform result
            (intensity or phase), which is shown in the screen
            or (optionally) saved to an image file.
            
        Technical note
        --------------
        The FFT results (intensity or phase) are shown/plotted using
        matplotlib. Therefore, many arguments of the current *show* method
        correspond to matplotlib parameters (such as *cmap* argument).
        '''
        
        # (0) Local function that adds colorbar not exceeding plot height
        # * this is surprisingly tricky => solution found in StackOverflow
        #   https://stackoverflow.com/q/18195758
        # * the usage of the function is slightly non-standard
        #   see the link above + search for: I created an IPython notebook
        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            '''Add a vertical color bar to an image plot.'''
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)
        
        # (1) Determine what to save
        # (default is FFT.intensity, but FFT.phase can be saved as well
        if what == 'intensity': arr = self.intensity
        else: arr = self.phase
        
        # (2) Set default colormap
        if cmap is None: cmap = 'gray'
        
        # (3) Plot 
        # Basic plot, saved to im - this is needed for (optional) colorbar
        im = plt.imshow(arr, cmap=cmap, vmax=icut)
        # Add nice colorbar not exceeding image height (using local function)
        if colorbar: add_colorbar(im)
 
        # (4) If saving to output file is required,
        # remove axes + edges and save the figure.
        if output is not None:
            plt.axis('off')
            plt.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0)
            
        # Show the figure.
        if axes == True: plt.axis('on')
        else: plt.axis('off') 
        plt.tight_layout()
        plt.show()
        
        
    def save(self, output='fft.png', what='intensity', 
             itype='16bit', icut=None, dpi=300):
        '''
        Save FFT object = Fourier transform of an image.

        Parameters
        ----------
        output : str or path-like object, optional, default is 'fft.png' 
            Name of the output file.
        what : str, optional, default is 'intensity'
            What result should be shown - 'intensity' or 'phase'.
        itype : str, optional, default is '16bit'
            Format of the output image.
            If '16bit' (default) => 16-bit grayscale image.
            If '8bit' (less suitable, narrow range) => 8-bit grayscale image.
        icut : int or float, optional, default is None
            Intensity cut value.
            If icut = 1000, all intenstity values >1000 are set to 1000.            
        dpi : int, optional, default is 300
            DPI of the saved image.

        Returns
        -------
        None
            The output is the image the Fourier transform result
            (intensity or phase), which is saved in {output} file.
        
        Technical note
        --------------
        The FFT results (images of 'intensity' or 'phase') can be saved
        either as matplotlib plots (show method with optional output argument)
        or standard grayscale images (save method = this method).
        The save method gives standard result, the show method can yield
        colour images with various cmaps and/or colorbars,
        which are suitable for presentations.
        '''
        
        # (1) Determine what to save
        # (default is FFT.intensity, but FFT.phase can be saved as well        
        if what == 'intensity': arr = self.intensity
        else: arr = self.phase
        
        # (2) Cut intenstity if required
        if icut is not None:
            arr = np.where(arr > icut, icut, arr)
            
        # (3) Rescale to 8bit or 16bit image
        if itype == '16bit': 
            norm_constant = 2**16 - 1 
            arr = arr/np.max(arr) * norm_constant
            arr = arr.astype(np.uint16)
        else: 
            norm_constant = 2**8 - 1
            arr = arr/np.max(arr) * norm_constant
            arr = arr.astype(np.uint8)
        
        # (4) Save array to Image using PIL
        # (saving with PIL = original number of points/pixels + selected dpi
        img = Image.fromarray(arr)
        img.save(output, dpi=(dpi,dpi)) 

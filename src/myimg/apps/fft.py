'''
Module: myimg.apps.fft
-----------------------

Fourier transform utilities for *MyImg* package.
'''

# Import modules ---------------------------------------------------------------

# System modules
import sys
from pathlib import Path

# Basic modules
import myimg.api
import scipy.fftpack

# Reading of input images
from PIL import Image

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1  # to add nice colorbars


# Define classes ---------------------------------------------------------------

class FFT:
    '''
    Class defining *FFT* object.
    '''


    def __init__(self, img, name=None):
        '''
        Initialize FFT object.

        * During the initialization, Fourier transform is calculated.
        * Simple illustration how the initialize and use FFT object follows.
        
        >>> # Example :: Calculate 2D-FFT of an image and show the result
        >>> import myimg.api as mi        # the standard import of myimg
        >>> mi.Apps.import_FFT_utils()    # import/add FFT utils to myimg.api
        >>> img = mi.MyImage('some.png')  # open an image using myimg.api
        >>> img_ft = fft.FFT(MyImage)     # calculate FFT of the {img} object
        >>> img_ft.show(cmap='magma')     # show the result

        Parameters
        ----------
        img : array or str/Path or MyImage object
            The 2D object, from which we will calculate FFT
            = 2D-DFFT = 2D discrete fast Fourier transform.
        name : str, optional, default is None
            Name of the input image (or array).
            If the input is an image or MyImage object,
            {name} is taken from it.

        Returns
        -------
        FFT object
            The object contains
            {FFT of input image} + {additional properties}.
        
        Technical details
        -----------------
        * FFT object saves the results of Fourier transform in three props:
            * {FFT.fft} = array of complex numbers
            * {FFT.intensity} = arr of intensities = magnitudes = real numbers
            * {FFT.phase} = array of phases = angles in range -pi:pi
        * FFT object is post-processed - all three above-listed arrays have
          the intensity center is shifted to the center of the array
          (using scipy.fftpack.fftshift).
        * FFT object may carry the information about name of the input image
          in property {FFT.name} on condition that:
            * It was created from an image or MyImage object. 
            * The optional argument {name} was used during initialization.
            * The property {FFT.name} property was set later.
        '''
        
        # Define local functions
        
        def get_name_from_image(img):
            # TODO
            return None
            
        def get_name_from_MyImage(img):
            # TODO
            return None
                
        # (0) Pre-initialize name of the image
        # * this simplifies further processing
        # * if the {name} they cannot be determined from args (img, name)
        #   then it is set to None - which is better than undefined value
        self.name = None
        
        # (1) Process the first argument = img: {array} or {image} or {MyImage}
        # * Props self.name and self.pixsize can be taken from {image / MyImage}
        # * This may influence the following processing step!
        if isinstance(img, np.ndarray):
            # img comes as numpy array
            # (the simplest case - just assign img to arr
            arr = img
        elif isinstance(img, str) or isinstance(img, Path):
            # image comes as str or Path => we assume it is an image name
            # TODO: check the image type
            img = Image.open(img)
            arr = np.asarray(img)
            # Try to et self.name from img = image name
            if name is None: self.name = get_name_from_image(img)
        elif isinstance(img, myimg.api.MyImage):
            # img comes as MyImage object
            # (check the image type and convert it to array if possible
            if img.itype in ('binary','gray','gray16'):
                arr = np.asarray(img.img)
                # Try to get name from img = MyImage object
                if name is None: self.name = get_name_from_MyImage(img)
            else:
                print('FFT works only for binary or grayscale images!')
                sys.exit()
        else:
            print('Unknown image type!')
            sys.exit()
        
        # (2) Process the other two optional arguments = name, pixsize
        # * the value of self.name might have been estimated
        #   from the image or MyImage in the previous step
        # * therefore, we re-define the value
        #   only if the optional arguments name was given
        #   (i.e. the argument - if given - has the priority
        if name is not None: self.name = name
        
        # (3) Calculate FFT, shift the origin, and save the results
        arr = scipy.fftpack.fft2(arr)
        arr = scipy.fftpack.fftshift(arr)
        self.fft = arr
        self.intensity = np.abs(arr)
        self.phase = np.angle(arr)
            
    
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
        None
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
        
        
    def save(self, output=None, what='intensity', 
             itype='16bit', icut=None, dpi=300):
        '''
        Save FFT object = Fourier transform of an image.

        Parameters
        ----------
        output : str or path-like object, optional, default is None. 
            Name of the output file.
            If output = None, then we will try output = self.name.
            If self.name = None, then we will set output = 'fft.png'.
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
        # (Why PIL? => original number of points/pixels + selected dpi
        # (a) find the name of output file
        if output is None:
            if self.name is not None:
                output = self.name
            else:
                output = 'fft.png'
        # (b) save the selected array (given by argument what) to file
        img = Image.fromarray(arr)
        img.save(output, dpi=(dpi,dpi))
        
    
    def save_with_extension(self):
        # TODO
        # 1. Build filename with extension.
        # 2. Call save  with name = filename_with_extension
        pass


class RadialProfile:
    '''
    Class defining *RadialProfile* object.

    Calculates mean intensity as a function of radius
    from a given center point.
    '''

    def __init__(self, img, center=None):
        # --- Lazy import for MyImage to avoid circular dependency
        try:
            from myimg.api import MyImage
        except ImportError:
            MyImage = None
    
        # (1) Convert input image to numpy array
        if isinstance(img, np.ndarray):
            arr = img
        elif MyImage is not None and isinstance(img, MyImage):
            if img.itype in ('binary', 'gray', 'gray16'):
                arr = np.asarray(img.img)
            else:
                print('Radial profiles only for binary or grayscale images!')
                sys.exit()
        elif isinstance(img, str):
            img = Image.open(img)
            arr = np.asarray(img)
        else:
            print('Unknown image type!')
            sys.exit()

        # (2) Determine center
        if center is None:
            x = arr.shape[0] / 2
            y = arr.shape[1] / 2
        else:
            x, y = center

        # (3) Calculate the radial profile
        R, I = RadialProfile.calc_radial(arr, center=(x, y))
        self.R = R
        self.I = I

    # -------------------------------------------------------------------------
    @staticmethod
    def calc_radial(arr, center):
        '''
        Calculate radial profile.
        
        * This method is employed during RadialProfile initialization.
        * It can be used alone, but this is not typical.
        '''
        
        # (1) Get image dimensions
        (height, width) = arr.shape

        # (2) Unpack center coordinates
        xc, yc = center

        # (3) Create distance map
        Y, X = np.meshgrid(np.arange(height) - yc,
                   np.arange(width) - xc,
                   indexing='ij')
        R = np.sqrt(X**2 + Y**2)

        # (4) Initialize variables
        radial_distance = np.arange(1, int(np.max(R)), 1)
        intensity = np.zeros(len(radial_distance))
        bin_size = 1

        # (5) Calculate mean intensity per radius bin
        for idx, r in enumerate(radial_distance):
            mask = (R >= r - bin_size) & (R < r + bin_size)
            values = arr[mask]
            intensity[idx] = np.mean(values) if values.size else 0

        return radial_distance, intensity

    # -------------------------------------------------------------------------
    def show(self, ax=None, **kwargs):
        '''
        Show the plot of radial profile.

        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        '''
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots()
            created_ax = True
    
        ax.plot(self.R, self.I, **kwargs)
        ax.set_xlabel("Radius [px]")
        ax.set_ylabel("Mean Intensity")
        ax.set_title("Radial Profile")
        ax.grid(True)
    
        if created_ax:
            plt.show()

    def save(self, filename="radial_profile.csv"):
        '''
        Save the radial profile as CSV file.
        '''
        import pandas as pd
        df = pd.DataFrame({"Radius_px": self.R,
                           "MeanIntensity": self.I})
        df.to_csv(filename, index=False)
        print(f"Radial profile saved to {filename}")


class AzimuthalProfile:
    '''
    Class defining *AzimuthalProfile* object.
    '''

    def __init__(self, img, center=None, bins=360):
        # --- Lazy import for MyImage to avoid circular dependency
        try:
            from myimg.api import MyImage
        except ImportError:
            MyImage = None
    
        # (1) Convert input image to numpy array
        if isinstance(img, np.ndarray):
            arr = img
        elif MyImage is not None and isinstance(img, MyImage):
            if img.itype in ('binary', 'gray', 'gray16'):
                arr = np.asarray(img.img)
            else:
                print('Azimuthal profiles only for binary/gray images!')
                sys.exit()
        elif isinstance(img, str):
            img = Image.open(img)
            arr = np.asarray(img)
        else:
            print('Unknown image type!')
            sys.exit()

        # (2) Determine center
        if center is None:
            x = arr.shape[0] / 2
            y = arr.shape[1] / 2
        else:
            x, y = center

        # (3) Calculate the azimuthal profile
        Theta, I = AzimuthalProfile.calc_azimuthal(
            arr, center=(x, y), bins=bins
        )
        self.Theta = Theta
        self.I = I

    # -------------------------------------------------------------------------
    @staticmethod
    def calc_azimuthal(arr, center, bins=360):
        '''
        Calculate azimuthal profile.
        
        * This method is employed during AzimuthalProfile initialization.
        * It can be used alone, but this is not typical.
        '''
        # (1) Get image dimensions
        (height, width) = arr.shape

        # (2) Unpack center coordinates
        xc, yc = center

        # (3) Create angle map
        Y, X = np.meshgrid(np.arange(height) - xc,
                           np.arange(width) - yc,
                           indexing='ij')
        theta = np.degrees(np.arctan2(Y, X)) % 360

        # (4) Bin intensities
        azimuthal_bins = np.linspace(0, 360, bins + 1)
        intensity = np.zeros(bins)

        for i in range(bins):
            mask = (theta >= azimuthal_bins[i]) & \
                   (theta < azimuthal_bins[i + 1])
            values = arr[mask]
            intensity[i] = np.mean(values) if values.size else 0

        theta_centers = (azimuthal_bins[:-1] + azimuthal_bins[1:]) / 2
        return theta_centers, intensity

    # -------------------------------------------------------------------------
    # For AzimuthalProfile
    def show(self, ax=None, **kwargs):
        '''
        Show the plot of aximuthal profile.

        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None
        '''
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots()
            created_ax = True
    
        ax.plot(self.Theta, self.I, **kwargs)
        ax.set_xlabel("Angle [deg]")
        ax.set_ylabel("Mean Intensity")
        ax.set_title("Azimuthal Profile")
        ax.grid(True)
    
        if created_ax:
            plt.show()
    
    def save(self, filename="azimuthal_profile.csv"):
        '''
        Save the azimuthal profile as CSV file.
        '''
        import pandas as pd
        df = pd.DataFrame({"Angle_deg": self.Theta,
                           "MeanIntensity": self.I})
        df.to_csv(filename, index=False)
        print(f"Azimuthal profile saved to {filename}")        

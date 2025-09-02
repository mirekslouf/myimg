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

    Calculates mean intensity as a function of angle
    (0–360 degrees) from a given center point.
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

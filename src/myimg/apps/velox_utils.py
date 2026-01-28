'''
Utilities for Velox EMD files

* Assumptions:
    - EMD files from Velox software (from Thermo Fisher)
    - EMD file names: {number} + {description} + {magnification/camera length}
* Limitations:
    - tested only for our Talos L120C microscope
    - detectors in our Talos: TEM(Ceta,SmartCam), STEM(Panther), HAADF, EDS
* Possible extensions:
    - modifications of the code may be necessary for other microscopes
    - contact the authors if you need to extend/modify/adjust this package
'''

# HyperSpy library
import hyperspy.api as hs
from hyperspy.signal import BaseSignal as hsBaseSignal

# Working with files
from pathlib import Path
import shutil
import itertools

# Working with tables
import pandas as pd
import tabulate

# Showing/saving images
import numpy as  np
import matplotlib.pyplot as plt
import PIL.Image

# Garbage collector - to release hsObjects from memory
import gc


class EMDobject:  
    '''
    Class providing EMDobjects.
    
    * EMDobject is a HyperSpy object with a few additional methods/properties.
    
    Technical notes
    ---------------
    
    EMDobject is build by means of three principal components.
    
    * Initialization:
        * __init__ initializes self.hsObject = HyperSpy object
        * EMD object = hsObject + self.additional_methods/props
    * Re-defined private methods: 
        * __getattr__ = delegates access to underlying HyperSpy object methods
        * __getitem__ = access to items (in case of list-of-HyperSpy objects)
        * __iter__ = access to iteration (in case of list-of-HyperSpy objects)
        * __len__ = access to len func (in case of list-of-HyperSpy objects)
    * Additional methods/props defined within this class, such as:
        * signal, describe, get_scale, save_signal2d ...
    '''
    
    def __init__(self, source_data):
        '''
        Initialize EMDobject.
        '''
        
        # (1) source_data = str => filename
        # => convert to Path and process below in step (2)
        if isinstance(source_data, str):
            source_data = Path(source_data)
        
        # (2) source_data = Path object => filename
        # => verify if the filename exist, open and save as self.hsObject
        if isinstance(source_data, Path):
            filename = source_data
            if not filename.exists(): 
                raise FileNotFoundError(f'File not found: {filename}')
            self.hsObject = hs.load(filename, lazy=True)
        
        # (3) source_data = hsBaseSignals (or perhaps list of hsBaseSignals)
        # => save as self.hsObject
        elif isinstance(source_data, hsBaseSignal) \
            or isinstance(source_data, list):   
            self.hsObject = source_data
        
        # (4) source_data is something else => error
        else: 
            raise Exception(f'Unknown source data: {source_data}.')


    def __getattr__(self, name):
        '''
        Delegate attribute access to the underlying HyperSpy object.
        
        Technical note
        --------------
        __getattr__ method ensures that
        unknonwn attributes/methods/properties are passed to self.hsObject
        '''
        return getattr(self.hsObject, name)
    
    
    def __getitem__(self, key):
        '''
        Delegate item access to the underlying HyperSpy object.
        
        Technical note
        --------------
        __getattr__ method does not give access to object ITEMS.
        '''
        # Get the requested item.
        result = self.hsObject[key]
        # Return the item as EMDobject
        # Reason: EMDobjects (not hsObjects) have access to added methods
        return EMDobject(result)
    
    
    def __iter__(self):
        '''
        Get iteration work - in case of list of HyperSpy objects.
        '''
        # We go though the items and return EMDobject for each item
        # Reason: EMDobjects (not hsObjects) have access to added methods
        for item in self.hsObject:
            yield EMDobject(item)


    def __len__(self):
        '''
        Get length of object - in case of list of HyperSpy objects.
        '''
        return len(self.hsObject)
    
    
    @property
    def signal(self):
        return EMDmetadata.signal_description(self)
    
    
    @property 
    def detectors(self):
        return EMDmetadata.list_of_detectors(self)
    
    
    @property
    def apertures(self):
        return EMDmetadata.list_of_apertures(self)
        
 
    def scale(self, out_format=1):
        '''
        Return scale from Velox EMD file.
        
        * Scale for 2Dsignals = images => pixel size.
        * Scale for 1Dsignals = spectra => unit on X-axis.
        * The information about scale is hidden in Velox EMD metadata.
        '''
        
        # (1) Get hsObject, which is saved in self.hsObject
        hsObject = self.hsObject
        
        # (2) If hsObject is a list of hsBaseSignals, take the first signal
        # (This may happen for multiple singal form BF, DF, HAADF...
        if isinstance(hsObject, list): hsObject = hsObject[0]
        
        # (3) Get pixel size from hsObject metadata
        number = hsObject.axes_manager.signal_axes[0].scale
        units = hsObject.axes_manager.signal_axes[0].units
        
        # (4) Return the result
        # (if compact = True, we return the result as single string
        if out_format == 0:
            return(number, units)
        elif out_format == 1:
            number = f'{number:.5g}'
            units = units.replace(' ','')
            number_and_units_as_str = ' '.join((number,units))
            return(number_and_units_as_str)
    
    
    def describe(self, verbose=0):
        '''
        Describe signals in the Velox EMD file, which was read into EMDobject.

        Parameters
        ----------
        verbose : int (0,1,2), optional, default is 0
            The argument determines, how much details we get. 

        Returns
        -------
        None
            The method just prints the information on the stdout.
        '''
        print('---')
        extra_space = ('') if verbose == 0 else ('   ')
        print(f'Signal {extra_space}: {EMDmetadata.signal_description(self)}')
        if verbose >= 1:
            print(f'Detectors : {EMDmetadata.list_of_detectors(self)}')
        if verbose >= 2:
            print(f'Apertures : {EMDmetadata.list_of_apertures(self)}')
        print('---')


    def signal2d_show(self, out_file=None, out_dpi=300, item=None, **kwargs):
        '''
        Show 2D-signal from the Velox EMD file.
        '''
        # (1) Get signal
        if item is None: 
            sig = self.data
        else:
            sig = self[item].data
        # (2) Plot signal
        plt.imshow(sig, **kwargs)
        plt.tight_layout()
        # (3) Save the plot if requested
        if out_file is not None:
            plt.savefig(out_file, dpi=out_dpi)
        # (4) Show the plot
        plt.show()

    
    def signal2d_save(self, out_file, item=None):
        '''
        Save 2D-signal from the Velox EMD file.
        '''
        # (1) Get signal
        if item is None: 
            sig = self.data
        else:
            sig = self[item].data
        # (2) Convert signal to numpy.array
        arr = sig.astype(np.uint16).compute()
        # (3) Convert array to image + save to {out_file}
        img = PIL.Image.fromarray(arr, mode='I;16')
        img.save(out_file)

        
    def close(self):
        '''
        Close EMDobject = release its links and prepare for destruction.
        
        Returns
        -------
        None
            The object is now empty.
            Its complete destruction must be done in the main code, 
            by calling `del(object)`.
        
        Technical notes
        ---------------
        * General rule: In Python, the objects don't kill themselves -
          they just drop what they own.
        * Therefore, this is just a *preparation* for object destruction,
          which must be done manually, in main program,
          using del(object).
        * Nevertheless, the object destruction may be useful
          in Spyder or Jupyter, where the opened EMD objects occupy memory
          and/or they may lock EMD files during repeated runs of a script.
        '''
        self.hsObject = None
        

class EMDfiles:
    '''
    Utilities to work with a group of Velox EMD files.
    
    This non-OO class contains two key functions:
        
    * rename = rename Velox EMD files (shorter names without whitespace)
    * describe = describe Velox EMD files (file, signal, apertures, detectors)
    
    Technical notes
    ---------------
    
    * The *rename* function renames also the exported files (PNG, TIFF...)
    * The *describe* function describes only the original EMD files.
    '''

        
    @classmethod
    def rename(cls, vdir, idigits=3, validate=False):
        cls.rename_orig(vdir, idigits, validate)
        cls.rename_exported(vdir, idigits, validate)
        
    
    @classmethod 
    def rename_orig(cls, vdir, idigits=3, validate=False):
        # If vdir is string, convert it to Path object.
        if isinstance(vdir, str): vdir = Path(vdir)
        # Define EMD files withing {vdir} directory.
        # (only Velox EMD files of type: 0001 - something.emd
        emd_files = vdir.rglob('???? - ?*.emd')
        # Go through EMD files, and rename them.
        for file in sorted(emd_files):
            new_name = cls.short_name(file, fullpath=True)
            if validate is True:
                print(f'{file} => {new_name}')
            else:
                shutil.move(file, new_name)

    
    @classmethod 
    def rename_exported(cls, vdir, idigits=3, validate=False):
        # If vdir is string, convert it to Path object.
        if isinstance(vdir, str): vdir = Path(vdir)
        # Define image files within {vdir} directory.
        # (only Velox exported files of type: 0001 - something.png|tif|txt
        img_files = itertools.chain(
            vdir.rglob('???? - ?*.png'),
            vdir.rglob('???? - ?*.tif'),
            vdir.rglob('???? - ?*.txt'))
        # Go through EMD files, and rename them.
        for file in sorted(img_files):
            new_name = cls.short_name( 
                file, fullpath=True, last_part_of_name=True)
            if validate is True:
                print(f'{file} => {new_name}')
            else:
                shutil.move(file, new_name)


    @classmethod
    def short_name(cls, file, 
                   idigits=3, last_part_of_name=False,
                   fullpath=False, subdirs=None, extension=False):
        # Convert file argument to Path object - for the sake of consistency
        if isinstance(file, str): file = Path(file)
        # Get stem of the filename
        # (this will be gradually changed to new_name
        orig_name = file.stem
        # Convert the initial separator between file number and the rest
        new_name = orig_name.replace(' - ','_')
        # Remove space between magnification and 'x'/'kx'
        new_name = new_name.replace(' x','x')
        new_name = new_name.replace(' kx','kx')
        # Remove space between camera-lenght and 'mm'
        new_name = new_name.replace(' mm','mm')
        # Correct possible error when the JobID is inserted as H66_
        new_name = new_name.replace('_ ','_')
        # Convert all other whitespace to underscore
        new_name = new_name.replace(' ','_')
        # Split to components
        all_elements = new_name.split('_')
        # New name contains only the first three elements
        name_elements = all_elements[0:3]
        # Add the last element (detector name) if requested
        if last_part_of_name:
            # We sould add the last element only if it exists!
            # (the name can have just 3 elements: {number} {jobID} {mag/CC}
            if len(all_elements) > 3:
                name_elements.append(all_elements[-1])
        # Redefine the number of digits in the first element
        name_elements[0] = f'{int(name_elements[0]):0{idigits}d}'
        # Combine all requested element
        new_name = '_'.join(name_elements)
        # Add subdirs and/or extension to the new_name if requested
        if fullpath is True:
            # Example: if fullpath = True => new_name = c:/d1/d2/new_name.emd
            new_name = Path(file.parent, new_name)        # add parent dir
            new_name = new_name.with_suffix(file.suffix)  # add extension
        elif (subdirs is not None) and (fullpath is False):
            # Assumption: file is c:/d1/d2/d3/orig_name.emd
            # Example: if subdirs = 2 => new_name = d2/d3/new_name
            new_name = cls.parent_dirs(file, subdirs) / new_name
        # Add extension of requeste
        if (extension is True) and (fullpath is False):
            # Example: if extension == True => new_name = new_name.emd
            # (NOT done if fullpath was requested =>  extension already added
            new_name = Path(new_name).with_suffix(file.suffix)
        # Return the shortened name
        return(new_name)
        
        
    @classmethod 
    def parent_dirs(path, n):
        # Get parent dir and split it to parts.
        # (input: c:\d1\d2\d3\file.txt => c:, d1, d2, d3
        parents = path.parent.parts
        # Get list of last n parent dirs, join them to path
        # (input {c:, d1, d2, d3} + {2} => d2\d3
        last_n_parents = Path(*parents[-n:])
        # Return the path of n last parent dirs
        return(last_n_parents)    


    @classmethod
    def describe(cls, vdir, fullpath=False, 
                 output_print=True, output_string=False, output_file=None):
        # If vdir is string, convert it to Path object.
        if isinstance(vdir, str): vdir = Path(vdir)
        # Prepare list/generator for emd_files
        emd_files = vdir.rglob('*.emd')
        # Prepare lists for the parameters we want to collect
        filenames = []
        signals   = []
        apertures = []
        detectors = []
        # Go through EMD files,
        # collect selected parameters and save them to lists.
        for file in emd_files:
            if fullpath is True:
                filenames.append(file)
            else:
                filenames.append(file.relative_to(vdir))
            hsObject = hs.load(file, lazy=True)
            signals.append( EMDmetadata.signal_description(hsObject) )
            apertures.append( EMDmetadata.list_of_apertures(hsObject) )
            detectors.append( EMDmetadata.list_of_detectors(hsObject) )
            # Release hsObject from memory
            del(hsObject)
        # Release hsObjects from memory
        gc.collect()
        # Convert the saved lists into a dataframe.
        df = pd.DataFrame({
            'Filename': filenames, 'Signal': signals,  
            'Apertures[um]': apertures, 'Detector(s)': detectors})
        # Convert the dataframe to a (nicely formatted) table.
        table = tabulate.tabulate(df,showindex=False, headers=df.columns)
        # Show/save final result
        # (1) If output_print=True (default) => print table to stdout.
        if output_print is True:
            print(table)
        # (2) If ouput_string=True (optional) => return table as string.
        if output_string is True:
            return(table)
        # (3) If output_file is specified (optional) => save table to file.
        if output_file is not None:
            with open(output_file, 'w') as fh: fh.writelines(table)    


class EMDmetadata:
    '''
    Access to important metadata of Velox EMD files.
    
    * This non-OO class defines several functions.
    * The functions are usually not used directly, but
      employed in the sister EMDobject and EMDfiles classes.
    '''
        
    # Class variables :: detector types
    TEM_detectors  = {'CETA', 'SMARTCAM', 'EAGLE'}
    STEM_detectors = {'BF', 'DF', 'HAADF'}
    EDS_detectors  = {'EDS', 'EDX'}

    
    # Class variable :: shorter names of selected detectors
    shorter_detector_names = {
        'BFS'          : 'BF',
        'DFS'          : 'DF',
        'DFS(0,1,2,3)' : 'DFi', 
        'DFS(4,5,6,7)' : 'DFo'}


    @staticmethod
    def is_single_signal(hsObject):
        test1 = isinstance(hsObject, hsBaseSignal)
        test2 = isinstance(hsObject, EMDobject) and len(hsObject) == 1
        is_single = test1 or test2
        return(is_single)
    
    @staticmethod 
    def is_list_of_signals(hsObject):
        test1 = isinstance(hsObject, list)
        test2 = isinstance(hsObject, EMDobject) and len(hsObject) > 1 
        is_list = test1 or test2
        return(is_list)


    @classmethod
    def signal_description(cls, hsObject):
        
        # (1) {hsObject} argument = datafile with one hsBaseSignal
        #     => describe the signal
        if EMDmetadata.is_single_signal(hsObject):
            # Determine detector_type (TEM or STEM or EDS)
            if cls.is_TEM(hsObject):
                detector_type = 'TEM'
            elif cls.is_STEM(hsObject):
                detector_type = 'STEM'
            elif cls.is_EDS_spectrum(hsObject):
                detector_type = 'EDS'
                return('EDS:spectrum')
            else:
                detector_type = 'Unknown'
            # Determine signal_type (image or diffractogram)
            if cls.is_Image(hsObject):
                signal_type = 'image'
            elif cls.is_Diffraction(hsObject):
                signal_type = 'diff'
            else:
                signal_type = 'Unknown'           
            return(f'{detector_type}:{signal_type}')
        
        # (2) {hsObject} argument = datafile with list of hsBaseSignals
        #     => describe just the first signal
        #     => assupmtion: all signals are analogous (typically BF,DF,HAADF)
        elif EMDmetadata.is_list_of_signals(hsObject):
            detector_type_and_signal = cls.signal_description(hsObject[0])
            return(detector_type_and_signal)
        
        # (3) {hsObject} is something else
        #     => return a brief message that this type of signal is unknown
        else: return "Uknown signal."


    @classmethod
    def list_of_detectors(cls, hsObject, concise=True):
        
        # (1) {hsObject} argument = datafile with one hsBaseSignal
        if EMDmetadata.is_single_signal(hsObject):
            sig  = hsObject.metadata.General.title
            sdim = hsObject.axes_manager.signal_dimension
            if concise:
                if sig in cls.shorter_detector_names.keys():
                    sig = cls.shorter_detector_names[sig]
                return(sig)
            else:
                return(f'{sdim}D:{sig}')
        
        # (2) {hsObject} argument = datafile with list of hsBaseSignals
        #     => describe the individual signals one by one 
        elif EMDmetadata.is_list_of_signals(hsObject):
            list_of_signals = [
                cls.list_of_detectors(sig, concise=concise)
                for sig in hsObject ]
            joined_list_of_signals = '[' + ' '.join(list_of_signals) + ']'
            return joined_list_of_signals
        
        # (3) {hsObject} is something else
        #     => return a brief message that this type of signal is unknown
        else: return "Unknown signal."


    @classmethod
    def list_of_apertures(cls, hsObject):
        
        # (1) {hsObject} argument = datafile with one hsBaseSignal
        if EMDmetadata.is_single_signal(hsObject):
            # Read apertures
            try:
                apertures = hsObject.original_metadata.Optics.Apertures
            except AttributeError:
                return "No apertures in original_metadata."
            # Go through all apertures + add them to list
            ap_list = []
            for ap_key in sorted(apertures.keys()):
                ap = apertures[ap_key]
                if ap.Type != 'None':
                    ap_diameter = round(float(ap.Diameter)*1e6)
                    ap_list.append(f'{ap.Name}:{ap_diameter:>3d}')
                else:
                    ap_list.append(f'{ap.Name}:---')
            # Convert the list to a final string with all apertures used
            ap_list = '[' + ' '.join(ap_list) + ']'
            return(ap_list)
        
        # (2) {hsObject} argument = datafile with list of hsBaseSignals
        elif EMDmetadata.is_list_of_signals(hsObject):
            ap_list = cls.list_of_apertures(hsObject[0])
            return(ap_list)
        
        # (3) {hsObject} is something else
        else: return('Unknown signal.')

            
    @classmethod
    def is_TEM(cls, hsObject):
        detector_name = hsObject.metadata.General.title
        # test is a genenerator object => easy to extend for new TEM detectors
        # generator => values one-by-one => any(test)==True if any value==True
        test = (k in detector_name.upper() for k in cls.TEM_detectors)
        return( any(test) )

    
    @classmethod
    def is_STEM(cls, hsObject):
        detector_name = hsObject.metadata.General.title
        # test is a genenerator object => easy to extend for new STEM detectors
        # generator => values one-by-one => any(test)==True if any value==True
        test = (k in detector_name.upper() for k in cls.STEM_detectors)
        return( any(test) )

    
    @classmethod
    def is_Image(cls, hsObject):
        units = hsObject.axes_manager.signal_axes[0].units
        # test is a genenerator object => easy to extend for other unit defs 
        # generator => values one-by-one => all(test)==True if all values==True
        test = (k not in units for k in ('1 /', '1/', 'eV'))
        return( all(test) )

    
    @classmethod
    def is_Diffraction(cls, hsObject):
        units = hsObject.axes_manager.signal_axes[0].units
        # test is a genenerator object => easy to extend for other unit defs 
        # generator => values one-by-one => all(test)==True if all values==True
        test = (k in units for k in ('1 /', '1/'))
        return( any(test) )

    
    @classmethod
    def is_EDS_spectrum(cls, hsObject):
        detector_name = hsObject.metadata.General.title
        # test is a genenerator object => easy to extend for new STEM detectors
        # generator => values one-by-one => any(test)==True if any value==True
        test = (k in detector_name.upper() for k in cls.EDS_detectors)
        return( any(test) )

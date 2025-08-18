'''
Package: MyImg
--------------

A toolbox for the processing of micrographs, which can do the following:
    
1. Process single micrographs (improve contrast, insert scalebars, etc.).
2. Prepare nice, publication-ready tiled images from processed micrographs.
3. Apply additional tools, such as: FFT, distributions, immunolabelling ...

See myimg.api for a simple user interface.

List of key objects, modules, and sub-packages:
    
* myimg.api = simple user interface, basic point to start
* myimg.objects = key objects used by myimg
    - myimg.objects.MyImage = single micrographs
    - myimg.objects.MyReport = multi-images = tiled images
* myimg.settings = default settings employed by MyImg objects
* myimg.apps = sub-package containing additional tools and/or applications
    - myimg.api.Apps = practical access to additional applications
    - myimg.api.Apps.FFT = sample additional application = Fourier transform
* myimg.utils = sub-package with code for specific/more complex utils in myimg
'''

__version__ = '0.3.6'

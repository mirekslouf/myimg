'''
Package: MyImg
--------------

A toolbox for the processing of micrographs, which can do the following:
    
1. Process single micrographs (improve contrast, insert scalebars, etc.).
2. Prepare publication-ready tiled images from the processed micrographs.
3. Apply additional tools, such as: FFT, size distributions, labelling ...

Key components:

* myimg.api = a simple user interface with basic tools
* myimg.apps = a subpackage providing access to the additional tools
'''

__version__ = '0.5.2'


# More complete list of objects, modules, and sub-packages:
# ---------------------------------------------------------    
# * myimg.api = simple user interface, the starting point
# * myimg.objects = key objects used by *MyImg*
#     - myimg.objects.MyImage = single micrographs
#     - myimg.objects.MyReport = multi-images = tiled images
# * myimg.apps = sub-package containing additional tools and/or applications
#     - myimg.apps = list of available additional applications
#     - myimg.api.Apps = practical access to additional applications
# * myimg.plots = simple module with auxiliary functions for plotting
# * myimg.utils = sub-package with code for specific/more complex utils in *MyImg*
# * myimg.settings = default settings employed by *MyImg* objects


# Obligatory acknowledgement -- the development was co-funded by TACR
# -------------------------------------------------------------------
#  TACR requires that the acknowledgement is printed when we run the program.
#  Nevertheless, Python packages run within other programs, not directly.
# The following code ensures that the acknowledgement is printed when:
#  (1) You run this file: __init__.py
#  (2) You run the package from command line: python -m ediff
# Technical notes:
#  To get item (2) above, we define __main__.py (next to __init__.py).
#  The usage of __main__.py is not very common, but still quite standard.

def acknowledgement():
    print('MyImg package - a toolbox for processing of microscopic images.')
    print('------')
    print('The development of the package was co-funded by')
    print('the Technology agency of the Czech Republic,')
    print('program NCK, project TN02000020.')
    
if __name__ == '__main__':
    acknowledgement()

'''
Subpackage myimg.apps
---------------------

The collection of additional applications for MyImage.

* myimg.apps.iLabels = process STEM images with nanoparticles
* myimg.apps.fft = FFT utilities = calculate and process 2D-DFFT of images
* myimg.apps.velox_utils = processing of Velox EMD files

The applications can be imported to myimg.api as follows:

>>> import myimg.api as mi                   # standard import of MyImg
>>> iLabels = myimg.Apps.import_iLabels()    # add iLabels package if needed
>>> fft = myimg.Apps.import_fft()            # add FFT processing if needed
>>> Velox = myimg.Apps.import_Velox_utils()  # add Velox EMD files procesing ...

More details can be found in the documentation
of the above-listed individual applications.
'''
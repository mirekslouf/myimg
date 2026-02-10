'''
Subpackage myimg.apps
---------------------

The collection of additional applications for *MyImg* package.

* myimg.apps.iLabels = process (S)TEM images with nanoparticle markers
* myimg.apps.MDistr = calculate distributions from microscopic datasets
* myimg.apps.fft = calculate and analyze Fourier transforms of images
* myimg.apps.velox_utils = processing of Velox EMD files

The applications can be imported to myimg.api as follows:

>>> import myimg.api as mi                   # standard import of MyImg
>>> iLabels = myimg.Apps.import_iLabels()    # add iLabels package if needed
>>> MDistr = myimg.Apps.import_MDistr()      # add MDistr package if needed
>>> fft = myimg.Apps.import_FFT_utils()      # add FFT processing if needed
>>> Velox = myimg.Apps.import_Velox_utils()  # add Velox EMD files procesing ...

More details can be found in the documentation
of the above-listed individual applications.
'''

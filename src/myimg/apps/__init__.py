'''
Subpackage myimg.apps
---------------------

The collection of additional applications for *MyImg* package.

* myimg.apps.MDistr = calculate distributions from microscopic datasets
* myimg.apps.fft_utils = calculate and analyze 1D and 2D Fourier transforms
* myimg.apps.iLabels = process (S)TEM images with nanoparticle markers
* myimg.apps.velox_utils = processing of Velox EMD files

The applications *can* be imported to myimg.api as follows:

>>> import myimg.api as mi                   # standard import of MyImg
>>> MDistr = myimg.Apps.import_MDistr()      # add size/shape distributions 
>>> fft = myimg.Apps.import_FFT_utils()      # add FFT calculation/processing
>>> iLabels = myimg.Apps.import_iLabels()    # add iLabels package
>>> Velox = myimg.Apps.import_Velox_utils()  # add Velox EMD files procesing

More details can be found in the documentation
of the above-listed individual applications.
'''

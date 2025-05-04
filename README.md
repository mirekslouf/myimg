MyImg :: Processing of micrographs
----------------------------------

* MyImg provides tools and apps for processing of microscopic images.
* Module MyImage  = process single image(s): contrast, label, scalebar ...
* Module MyReport = create image report (several images in a rectangular grid)
* Additional apps/sub-packages for: FFT, size distributions, imunolabelling ...

Principle
---------

* TODO

Installation
------------

* Requirement: Python with sci-modules: numpy, matplotlib, pandas
* `pip install scikit-image` = additional package for image processing 
* `pip install myimg` = MyImg package itself (uses all packages above)

Quick start
-----------

* [Example 1](https://www.dropbox.com/scl/fi/uv76nx5e78ck3ir4vv6zh/ex1_one-migrograph.nb.html.pdf?rlkey=mzsxovdriljt1054tpjey8kpm&st=dqjt2w0c&dl=0)
  :: MyImage :: process single image(s)
* [Example 2]()
  :: MyReport :: create nice, publication-ready image reports
* [Example 3]()
  :: Apps/FFT :: calculate Fourier transforms and use them for image analysis
 

Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/myimg) repository.
* [GitHub](https://github.com/mirekslouf/myimg) repository.
* [GitHub Pages](https://mirekslouf.github.io/myimg)
  with [documentation](https://mirekslouf.github.io/myimg/docs). 

Versions of MyImg
-----------------

* Version 0.1 = 1st draft, too complex, later completely re-written 
* Version 0.2 = 2nd draft, better concept; functions: cut, crop, label, scalebar
* Version 0.3 = MyImage and MyReport modules Ok; FFT and iLabels semi-finished 
* Version 0.4 = TODO: scalebar/stripes,border/shadow, improved FFT, iLabels ...

MyImg :: Processing of micrographs
----------------------------------

* MyImg contains tools/modules for the processing of microscopic images.
* Module MyImage = improve image(s): contrast, label, caption, scalebar ...
* Module Montage = create image montages (several images in a rectangular grid)
* Additional utilities/modules for: FFT, imunolabelling ...

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

* [Example 1](https://www.dropbox.com/scl/fi/uv76nx5e78ck3ir4vv6zh/ex1_one-migrograph.nb.html.pdf?rlkey=mzsxovdriljt1054tpjey8kpm&st=dqjt2w0c&dl=0):
  one micrograph, improve contrast, insert scalebar ...
* Example 2:
  multiple micrographs, prepare a publication-ready image.

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
* Version 0.3 = MyImage and Montage modules fully working
* Version 0.4 = TODO: scalebar/stripes, border/shadow, iLabels ...

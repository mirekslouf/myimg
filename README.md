MyImg :: Processing of micrographs
----------------------------------

* MyImg is a toolbox for the processing of micrographs, which can:
	1. Process single micrographs (improve contrast, insert scalebars, etc.).
	2. Prepare publication-quality tiled images from the processed micrographs.
	3. Run additional applications such as:
		- FFT = 2D Fourier transform utilities
		- MDistr = size distributions from series of micrographs
		- iLabels = find and categorize nanoparticle markers in (S)TEM images


Principle
---------

* TODO


Installation
------------

* Requirement: Python with sci-modules: numpy, matplotlib, pandas
* `pip install scikit-image` = additional package for image processing
* `pip install hyperspy[all]` = package that can read Velox EMD files
* `pip install exspy[all]` = supplement to hyperspy package
* `pip install tabulate` = tabulate module for nice outputs
* `pip install myimg` = MyImg package itself (uses all packages above)


Quick start
-----------

* [Worked example](https://drive.google.com/file/d/1abxWcKD9GOGtMYASjO67RaUsUQfJN8DO/view?usp=sharing)
  shows a simple use of MyImg package.
* [Help on GitHub](https://mirekslouf.github.io/myimg/docs/)
  with complete
  [package documentation](https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html)
  and
  [additional examples](https://drive.google.com/drive/folders/1ylURF2U1EB3gdyUug_uOTFbfE0ASZX7n?usp=sharing).

 
Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/myimg) repository -
  the stable version to install.
* [GitHub](https://github.com/mirekslouf/myimg) repository - 
  the current version under development.
* [GitHub Pages](https://mirekslouf.github.io/myimg/) -
  the more user-friendly version of GitHub website.
 


Versions of MyImg
-----------------

* Version 0.1 = 1st draft: too complex, later completely re-written 
* Version 0.2 = 2nd draft: MyImage object with cut, crop, label, scalebar
* Version 0.3 = objects: MyImage, MyReport; apps: FFT, iLabels (semi-finished)
* Version 0.4 = apps.velox: utilities to process Velox EMD files
* Version 0.5 = better Apps interface + improved Apps + updated documentation
* Version 0.6 = TODO: add scalebar-stripes, mdistr + finalize FFT, iLabels

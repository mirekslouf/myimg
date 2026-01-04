MyImg :: Processing of micrographs
----------------------------------

* MyImg is a toolbox for the processing of micrographs, which can:
	1. Process single micrographs (improve contrast, insert scalebars, etc.).
	2. Prepare high-quality tiled images from the processed micrographs.
	3. Run additional applications such as:
		- FFT = 2D Fourier transforms
		- MDistr = size distributions from series of micrographs
		- iLabels = find and categorize nanoparticles on (S)TEM micrographs


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

* Jupyter notebooks with comments:
	- [Example 1](https://www.dropbox.com/scl/fi/0vq7pcrna6v3qqxcjg7zr/ex1_single-images.nb.html.pdf?rlkey=z9ft9iapz8zm8kdurxs4kjqia&st=g7x2zuwx&dl=0)
      :: MyImage :: process single image(s)
	- [Example 2](https://www.dropbox.com/scl/fi/x9nvbqr2epd2fms8k1qx8/ex2_tiled-images.nb.html.pdf?rlkey=qcjx8tcv3pjoxgs4kkjplo61m&st=ylwaxak1&dl=0)
	  :: MyReport :: create nice, publication-ready image reports
	- [Example 3](https://www.dropbox.com/scl/fi/u4brdkvufnb4hxn9p3z1w/ex3_2d-fft.nb.html.pdf?rlkey=sus0snlqfgmpvbtol2xojqizu&st=80ela7q3&dl=0)
	  :: Apps/FFT :: calculate and analyze 2D Fourier transforms
* Complete set of examples including testing data at
  [DropBox](https://www.dropbox.com/scl/fo/rdnhfl0eaiv3yueze2b24/APLqQqVV8BG8XC1_VDPbFxY?rlkey=pdzjibm35609oxtgfinxls3ga&st=qj8ul380&dl=0).
 
Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/myimg) repository.
* [GitHub](https://github.com/mirekslouf/myimg) repository.
* [GitHub Pages](https://mirekslouf.github.io/myimg/)
  with [help](https://mirekslouf.github.io/myimg/docs)
  and [complete package documentation](https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html). 


Versions of MyImg
-----------------

* Version 0.1 = 1st draft: too complex, later completely re-written 
* Version 0.2 = 2nd draft: MyImage object with cut, crop, label, scalebar
* Version 0.3 = objects: MyImage, MyReport; apps: FFT, iLabels (semi-finished)
* Version 0.4 = apps.velox: utilities to process Velox EMD files 
* Version 0.5 = TODO: add scalebar-stripes, mdistr + finalize FFT, iLabels

MyImg :: Processing of micrographs
----------------------------------

* *MyImg* is a toolbox for the processing of micrographs, <br>
  which improves single images, batch-processes multiple images, and much more. 
* If you use *MyImg* in your research, <br>
  please cite recent paper: [*Microscopy and Microanalysis* 31, 2025, ozaf045.](https://doi.org/10.1093/mam/ozaf045)


Principle
---------
1. [Single images](https://mirekslouf.github.io/myimg/docs/assets/1_myimg.png)
   = inserting scalebars, auto-adjusting contrast ...
2. [Multiple images](https://mirekslouf.github.io/myimg/docs/assets/2_myreport.png)
   = batch processing, publication-quality tiled images ...
3. [Additional applications](https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg/apps.html)
   = Fourier transforms, particle size distributions  ...
   <br>
   <br>
   <img src="https://mirekslouf.github.io/myimg/docs/assets/1_myimg.png" alt="MyImg principle" width="600"/>
	
Installation
------------

* MyImg installation: `pip install myimg` 
* MyImg additional packages, which are auto-installed if not present: 
	- `numpy`, `matplotlib`, `pandas`
	- `scikig-image`, `hyperspy[all]`, `exspy[all]`, `tabulate`

Quick start
-----------

* [Worked example](https://drive.google.com/file/d/1abxWcKD9GOGtMYASjO67RaUsUQfJN8DO/view?usp=sharing)
  shows *MyImg* package in action.
* [Help on GitHub](https://mirekslouf.github.io/myimg/docs/)
  with
  [complete documentation](https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html)
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
* Version 0.6 = TODO: finalize apps.fft_utils, apps.iLabels

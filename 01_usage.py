import myimg.api as mi

# (1) Read image and adjust it
# (this is working in current version
img = mi.MyImage('imParticles.png', peaks=True)
#img.cut(60)

# (2) Read, search, correct and classify peaks
# (this is to be programmed

# (a) Read peaks from file
# (TXT file, read/saved using pandas
img.peaks.read('pdParticles.pkl')
img.peaks.show_as_text()
img.peaks.show_in_image()
img.peaks.save_with_extension('_p.txt')

# (b) Search peaks
# (methods: 'manual', 'threshold', ...)
img.peaks.find(method='manual')
img.peaks.show_as_text()
img.peaks.show_in_image()
img.peaks.save_with_extension('_p.txt')

# (c) Correct peaks
# (method: 'manual' - probably the only option
img.peaks.correct(method='manual')
img.peaks.show_as_text()
img.peaks.show_in_image()
img.peaks.save_with_extension('_pc.txt')

# (d) Classify peaks
# (method: 'gauss_fit' - for now
img.peaks.classify(method='gauss_fit')
img.peaks.show_as_text()
img.peaks.show_in_image()
img.peaks.save_with_extension('_pcc.txt')

# This is how the large method can be "outsourced"
# (idea: img.peaks.search = wrapper/container for mi.utils.peaks.search ...
# (the same idea has already been realized for img.label, img.scalebar ...
mi.utils.peaks.search(img, method='manual')
mi.utils.peaks.classify(img, method='gauss_fit')

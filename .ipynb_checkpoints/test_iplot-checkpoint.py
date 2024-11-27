# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:18:42 2024

@author: Jakub
"""
import myimg.api as mi

# (1) Read image and adjust it
# (this is working in current version
img = mi.MyImage('imParticles.png', peaks=True)
#img.cut(60)

# (2) Read, search, correct and classify peaks


# (a) Read peaks from file
# (TXT file, read/saved using pandas
img.peaks.read('pdParticles.pkl')
img.peaks.show_as_text()
img.peaks.show_in_image()


# (b) Search peaks
# (methods: 'manual', 'threshold', ...)
img.peaks.find(method='manual')
img.peaks.save_with_extension('_p.txt')
#!/usr/bin/env python

import sys
import numpy as np

from   setuptools import setup, find_packages

setup(name             = 'hist_gpu',
      version          = 'v0.1',
      description      = 'Hists',,
      author_email     = 'i.khrykin@gmail.com',
      url              = 'https://github.com/ikhrykin/hist_gpu',
      packages         = find_packages( include=['*.py'] ),
      install_requires = [ 'astropy',
                           'numpy',
                           'scipy',
                           'matplotlib'
                         ])

print ('********************************************************************')
print ('HIST_GPU package has been succsefully installed to your machine. Enjoy!')
print ('********************************************************************')
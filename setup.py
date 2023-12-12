#!/usr/bin/env python

import sys
import numpy as np

from   setuptools import setup, find_packages

setup(name             = 'hist_gpu',
      author_email     = 'i.khrykin@gmail.com',,
      packages         = find_packages( include=['*.py'] ) )

print ('********************************************************************')
print ('package has been succsefully installed to your machine. Enjoy!')
print ('********************************************************************')
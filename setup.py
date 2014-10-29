#!/usr/bin/env python
# -*- coding: utf-8

'''
Setup script.
'''

import glob
import numpy
import os
import sys

from distutils.core import setup
from distutils.extension import Extension

SOURCE = '.'

if sys.version_info[:2] < (2, 6):
    print('Requires Python version 2.7 or later (%d.%d detected).' %
          sys.version_info[:2])
    sys.exit(-1)

def get_packages():
    '''Appends all packages (based on recursive sub dirs)'''

    packages  = ['klkmeans']
    return_val = []
    while len(packages) > 0:
        package = packages.pop(0)
        return_val.append(package)
        base = os.path.join(package, '**/')
        sub_dirs = glob.glob(base)

        while len(sub_dirs) != 0:
            for sub_dir in sub_dirs:
                package_name = sub_dir.replace('/', '.')
                if package_name.endswith('.'):
                    package_name = package_name[:-1]

                packages.append(package_name)
        
            base = os.path.join(base, '**/')
            sub_dirs = glob.glob(base)

    return return_val

if __name__ == "__main__":
    os.chdir(SOURCE)
    packages = get_packages()
    
    setup(name         = 'klkmeans',
          packages     = packages)

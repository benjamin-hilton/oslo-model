from distutils.core import setup, Extension
import numpy as np

module1 = Extension('oslo',
                    sources=['oslomodule.cpp'],
                    extra_compile_args=['-std=c++11', '-O3'])

setup(name='OsloModule',
      version='0.1',
      description='Implements the Oslo Model.',
      include_dirs=[np.get_include()],
      ext_modules=[module1])

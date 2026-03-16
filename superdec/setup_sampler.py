from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([
        Extension(
            "superdec.fast_sampler._sampler",
            [
                "superdec/fast_sampler/_sampler.pyx",
                "superdec/fast_sampler/sampling.cpp"
            ],
            language="c++",
            libraries=["stdc++"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++17", "-O3"],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )
    ],
    language_level=3),
    zip_safe=False
)

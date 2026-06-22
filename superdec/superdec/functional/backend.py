# https://github.dev/mit-han-lab/pvcnn/blob/master/modules/functional/backend.py

import os
import shutil
from torch.utils.cpp_extension import load

# CUDA 12.8 supports up to GCC 14.  If the system default is newer (e.g. GCC
# 15 on Ubuntu 25.04+), pass an older version as -ccbin via CC so PyTorch's
# JIT build picks it up.
_extra_cuda_cflags = []
if 'CC' not in os.environ:
    for _gcc in ('gcc-14', 'gcc-13', 'gcc-12'):
        _path = shutil.which(_gcc)
        if _path:
            os.environ['CC'] = _path
            break

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']
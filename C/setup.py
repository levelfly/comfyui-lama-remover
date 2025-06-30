# /C/setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    # 模組名稱，這個名稱將在 Python 中被 import
    name='lama_cpp',
    
    ext_modules=[
        CUDAExtension(
            # 擴充模組的完整名稱，我們將 import lama_cpp._C
            name='lama_cpp._C',  
            
            # 需要編譯的原始檔案列表
            sources=[
                'gaussian_blur.cpp',
                'gaussian_blur_kernel.cu',
            ],
            
            # 為編譯器加入一些額外的參數，可以提升效能
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    
    # 這個指令會告訴 setuptools 如何執行編譯
    cmdclass={
        'build_ext': BuildExtension
    },
    
    # 將 'lama_cpp' 這個名稱定義為一個可被 import 的包
    packages=['lama_cpp']
)
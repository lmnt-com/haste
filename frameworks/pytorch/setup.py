
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='ligru_cuda',
#     ext_modules=[
#         CUDAExtension('ligru_cuda', [
#             '/home/adel/Documents/haste/frameworks/pytorch/ligru.cc',
#             '/home/adel/Documents/haste/frameworks/pytorch/support.cc',
#             '/home/adel/Documents/haste/lib/ligru_forward_gpu.cu.cc',
#             '/home/adel/Documents/haste/lib/haste.h',
#             '/home/adel/Documents/haste/frameworks/pytorch/support.h'
#         ]),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })
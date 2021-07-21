# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys

from glob import glob
from platform import platform
from torch.utils import cpp_extension
from setuptools import setup
from setuptools.dist import Distribution


class BuildHaste(cpp_extension.BuildExtension):
  def run(self):
    os.system('make haste')
    super().run()


base_path = os.path.dirname(os.path.realpath(__file__))
if 'Windows' in platform():
  CUDA_HOME = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
  extra_args = []
else:
  CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
  extra_args = ['-Wno-sign-compare']

with open(f'frameworks/pytorch/_version.py', 'wt') as f:
  f.write(f'__version__ = "{VERSION}"')

extension = cpp_extension.CUDAExtension(
    'haste_pytorch_lib',
    sources = glob('frameworks/pytorch/*.cc'),
    extra_compile_args = extra_args,
    include_dirs = [os.path.join(base_path, 'lib'), os.path.join(CUDA_HOME, 'include')],
    libraries = ['haste'],
    library_dirs = ['.'])

setup(name = 'haste_pytorch',
    version = VERSION,
    description = DESCRIPTION,
    long_description = open('README.md', 'r',encoding='utf-8').read(),
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    license = LICENSE,
    keywords = 'pytorch machine learning rnn lstm gru custom op',
    packages = ['haste_pytorch'],
    package_dir = { 'haste_pytorch': 'frameworks/pytorch' },
    install_requires = [],
    ext_modules = [extension],
    cmdclass = { 'build_ext': BuildHaste },
    classifiers = CLASSIFIERS)

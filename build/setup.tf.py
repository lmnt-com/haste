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

from setuptools import setup
from setuptools.dist import Distribution
from distutils.command.build import build as _build


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""
  def has_ext_modules(self):
    return True


class BuildHaste(_build):
  def run(self):
    os.system('make libhaste_tf')
    super().run()


with open(f'frameworks/tf/_version.py', 'wt') as f:
  f.write(f'__version__ = "{VERSION}"')

setup(name = 'haste_tf',
    version = VERSION,
    description = DESCRIPTION,
    long_description = open('README.md', 'r').read(),
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    license = LICENSE,
    keywords = 'tensorflow machine learning rnn lstm gru custom op',
    packages = ['haste_tf'],
    package_dir = { 'haste_tf': 'frameworks/tf' },
    package_data = { 'haste_tf': ['*.so'] },
    install_requires = [],
    zip_safe = False,
    distclass = BinaryDistribution,
    cmdclass = { 'build': BuildHaste },
    classifiers = CLASSIFIERS)

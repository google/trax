# coding=utf-8
# Copyright 2021 The Trax Authors.
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

# coding=utf-8
"""Install trax."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='trax',
    version='1.4.1',
    description='Trax',
    long_description=(
        'Trax helps you understand deep learning. We start with basic maths and'
        ' go through layers, models, supervised and reinforcement learning. We '
        'get to advanced deep learning results, including recent papers and '
        'state-of-the-art models.'
    ),
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/google/trax',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'funcsigs',
        'gin-config',
        'gym',
        'jax',
        'jaxlib',
        'matplotlib',
        'numpy',
        'psutil',
        'scipy',
        'six',
        'tensorflow-datasets',
        'tensorflow-text',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.15.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.15.0'],
        't5': ['t5>=0.4.0'],
        'tests': [
            'attrs',
            'jupyter',
            'mock',
            'parameterized',
            'pylint',
            'pytest',
            'wrapt==1.11.*',
        ],
        't2t': ['tensor2tensor',],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning jax',
)

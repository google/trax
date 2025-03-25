# coding=utf-8
# Copyright 2022 The Trax Authors.
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
    name="trax",
    version="1.5.1",
    description="Trax",
    long_description=(
        "Trax helps you understand deep learning. We start with basic maths and"
        " go through layers, models, supervised and reinforcement learning. We "
        "get to advanced deep learning results, including recent papers and "
        "state-of-the-art models."
    ),
    author="Google Inc.",
    author_email="no-reply@google.com",
    url="http://github.com/google/trax",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "absl-py==1.4.0",
        "funcsigs==1.0.2",
        "gin-config==0.5.0",
        "gym==0.26.2",
        "jax==0.4.20",
        "jaxlib==0.4.20",
        "matplotlib==3.8.0",
        "numpy==1.23.5",
        "psutil==5.9.5",
        "scipy==1.11.3",
        "six==1.14.0",
        "tensorflow-datasets==4.2.0",
        "tensorflow-text==2.13.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.13.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=2.13.0"],
        "t5": ["t5==0.9.2"],
        "tests": [
            "attrs==23.1.0",
            "jupyter",
            "mock==5.1.0",
            "parameterized==0.9.0",
            "pylint==2.17.7",
            "pytest==7.4.2",
            "wrapt==1.15.0",
        ],
        "t2t": [
            "tensor2tensor==1.15.7",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="tensorflow machine learning jax",
)

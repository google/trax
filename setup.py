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

from setuptools import find_packages, setup

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
        "absl-py==2.2.0",
        "funcsigs",
        "gin-config==0.5.0",
        "gym",
        "jax==0.5.3",
        "jaxlib==0.5.3",
        "matplotlib",
        "numpy==2.0.2",
        "psutil==7.0.0",
        "scipy==1.15.2",
        "six==1.14.0",
        "tensorflow-datasets==4.9.8",
        "tensorflow-text==2.17.0",
        "tensorflow-estimator==2.15.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.17.0"],
        "tensorflow_cuda": ["tensorflow[and-cuda]==2.17.0"],
        "t5": ["t5==0.9.4"],
        "tests": [
            "attrs==25.3.0",
            "jupyter",
            "mock==5.1.0",
            "parameterized==0.9.0",
            "pylint==3.3.6",
            "pytest==8.3.5",
            "wrapt==1.17.2",
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

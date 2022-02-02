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

#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"

# Make sure we have the latest pip and setuptools installed.
pip install -q -U pip
pip install -q -U setuptools

# Make sure we have the latest version of numpy - avoid problems we were
# seeing with Python 3
pip install -q -U numpy

# Install appropriate version to tensorflow.
if [[ "$TF_VERSION" == "tf-nightly"  ]]
then
  pip install tf-nightly;
else
  pip install -q "tensorflow==$TF_VERSION"
fi

# Just print the version again to make sure.
python -c 'import tensorflow as tf; print(tf.__version__)'

# First ensure that the base dependencies are sufficient for a full import
pip install -q -e .

# Then install the test dependencies
pip install -q -e .[tests]
# Make sure to install the atari extras for gym
pip install "gym[atari]"

# Coverage.
pip install coverage coveralls

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

GIT_COMMIT_ID=${1:-""}
[[ -z $GIT_COMMIT_ID ]] && echo "Must provide a commit" && exit 1

TMP_DIR=$(mktemp -d)
pushd $TMP_DIR

echo "Cloning trax and checking out commit $GIT_COMMIT_ID"
git clone https://github.com/google/trax.git
cd trax
git checkout $GIT_COMMIT_ID

python3 -m pip install wheel twine pyopenssl

# Build the distribution
echo "Building distribution"
python3 setup.py sdist
python3 setup.py bdist_wheel --universal

# Publish to PyPI
echo "Publishing to PyPI"
python3 -m twine upload dist/*

# Cleanup
rm -rf build/ dist/ trax.egg-info/
popd
rm -rf $TMP_DIR

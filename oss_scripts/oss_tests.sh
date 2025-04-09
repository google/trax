#!/bin/bash

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

set -v  # print commands as they're executed

# aliases aren't expanded in non-interactive shells by default.
shopt -s expand_aliases

# Instead of exiting on any failure with "set -e", we'll call set_status after
# each command and exit $STATUS at the end.
STATUS=0
function set_status() {
    local last_status=$?
    if [[ $last_status -ne 0 ]]
    then
      echo "<<<<<<FAILED>>>>>> Exit code: $last_status"
    fi
    STATUS=$(($last_status || $STATUS))
}

# Check env vars set
echo "${TRAX_TEST:?}"
set_status
if [[ $STATUS -ne 0 ]]
then
  exit $STATUS
fi

# Check import.
python -c "import trax"
set_status

# # Run pytest with coverage.
# alias pytest='coverage run -m pytest'

# Check tests, check each directory of tests separately.
if [[ "${TRAX_TEST}" == "lib" ]]
then
  echo "Testing all framework packages..."

  ## Core Trax and Supervised Learning
  pytest tests/data
  set_status

  pytest tests/fastmath
  set_status

  pytest tests/layers
  set_status

  pytest tests/learning
  set_status

  pytest tests/models
  set_status

  pytest tests/optimizers
  set_status

  pytest tests/tf
  set_status

  pytest tests/trainers
  set_status

  pytest tests/utils/import_test.py
  set_status

  pytest tests/utils/shapes_test.py
  set_status

else
  echo "No testing ..."
  # Models, RL and misc right now.

  # Check notebooks.

  # TODO(afrozm): Add more.
  #  jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 \
  #    --ExecutePreprocessor.timeout=600 --to notebook --execute \
  #    trax/intro.ipynb;
  #  set_status
fi

exit $STATUS

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

# Check tests, separate out directories for easy triage.

if [[ "${TRAX_TEST}" == "lib" ]]
then
  ## Core Trax and Supervised Learning

  # Disabled the decoding test for now, since it OOMs.
  # TODO(afrozm): Add the decoding_test.py back again.

  # training_test and trainer_lib_test parse flags, so can't use with --ignore
  pytest \
    --ignore=trax/supervised/callbacks_test.py \
    --ignore=trax/supervised/decoding_test.py \
    --ignore=trax/supervised/decoding_timing_test.py \
    --ignore=trax/supervised/trainer_lib_test.py \
    --ignore=trax/supervised/training_test.py \
    trax/supervised
  set_status

  # Testing these separately here.
  pytest \
    trax/supervised/callbacks_test.py \
    trax/supervised/trainer_lib_test.py \
    trax/supervised/training_test.py
  set_status

  pytest trax/data
  set_status

  # Ignoring acceleration_test's test_chunk_grad_memory since it is taking a
  # lot of time on OSS.
  pytest \
    --deselect=trax/layers/acceleration_test.py::AccelerationTest::test_chunk_grad_memory \
    --deselect=trax/layers/acceleration_test.py::AccelerationTest::test_chunk_memory \
    --ignore=trax/layers/initializers_test.py \
    --ignore=trax/layers/test_utils.py \
    trax/layers
  set_status

  pytest trax/layers/initializers_test.py
  set_status

  pytest trax/fastmath
  set_status

  pytest trax/optimizers
  set_status

  # Catch-all for futureproofing.
  pytest \
    --ignore=trax/trax2keras_test.py \
    --ignore=trax/data \
    --ignore=trax/fastmath \
    --ignore=trax/layers \
    --ignore=trax/models \
    --ignore=trax/optimizers \
    --ignore=trax/rl \
    --ignore=trax/supervised \
    --ignore=trax/tf_numpy
  set_status
else
  # Models, RL and misc right now.

  ## Models
  # Disabled tests are quasi integration tests.
  pytest \
    --ignore=trax/models/reformer/reformer_e2e_test.py \
    --ignore=trax/models/reformer/reformer_memory_test.py \
    --ignore=trax/models/research/terraformer_e2e_test.py \
    --ignore=trax/models/research/terraformer_memory_test.py \
    --ignore=trax/models/research/terraformer_oom_test.py \
    trax/models
  set_status

  ## RL Trax
  pytest trax/rl
  set_status

  ## Trax2Keras
  # TODO(afrozm): Make public again after TF 2.5 releases.
  # pytest trax/trax2keras_test.py
  # set_status

  # Check notebooks.

  # TODO(afrozm): Add more.
  jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 --to notebook --execute \
    trax/intro.ipynb;
  set_status
fi

# TODO(traxers): Test tf-numpy separately.

exit $STATUS

# Copyright 2020 The Trax Authors.
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
echo "${TF_VERSION:?}" && \
echo "${TF_LATEST:?}" && \
echo "${TRAVIS_PYTHON_VERSION:?}"
set_status
if [[ $STATUS -ne 0 ]]
then
  exit $STATUS
fi

# Check import.
python -c "import trax"
set_status

# Check notebooks.
# TODO(afrozm): Add more.
jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 --to notebook --execute \
  trax/intro.ipynb;
set_status

# Check tests, separate out directories for easy triage.
pytest --disable-warnings trax/layers
set_status

pytest --disable-warnings trax/math
set_status

pytest --disable-warnings trax/models
set_status

pytest --disable-warnings trax/optimizers
set_status

pytest --disable-warnings trax/rl
set_status

pytest --disable-warnings trax/supervised
set_status

pytest --disable-warnings \
  --ignore=trax/layers \
  --ignore=trax/math \
  --ignore=trax/models \
  --ignore=trax/optimizers \
  --ignore=trax/rl \
  --ignore=trax/supervised
set_status

exit $STATUS

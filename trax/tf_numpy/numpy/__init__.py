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

"""NumPy like wrapper for Tensorflow."""

# pylint: disable=wildcard-import
# pylint: disable=g-import-not-at-top
# pylint: disable=g-direct-tensorflow-import

try:
  # Note that this import will work in tf-nightly and TF versions 2.4 and
  # higher.
  from tensorflow.experimental.numpy import *
  # TODO(agarwal): get rid of following imports.
  from tensorflow.experimental.numpy import random
  from tensorflow import bfloat16
  import numpy as onp
  from tensorflow.python.ops.numpy_ops.np_dtypes import canonicalize_dtype
  from tensorflow.python.ops.numpy_ops.np_dtypes import default_float_type
  from tensorflow.python.ops.numpy_ops.np_dtypes import is_allow_float64
  from tensorflow.python.ops.numpy_ops.np_dtypes import set_allow_float64

  random.DEFAULT_RANDN_DTYPE = onp.float32
except ImportError:
  try:
    # Note that this import will work in TF 2.3 and higher.
    from tensorflow.python.ops.numpy_ops import *
    from tensorflow import bfloat16

  except ImportError:
    # Note that this fallback will be needed for TF 2.2.
    from tensorflow import newaxis

    from trax.tf_numpy.numpy_impl import random

    # pylint: disable=wildcard-import
    from trax.tf_numpy.numpy_impl.array_ops import *
    from trax.tf_numpy.numpy_impl.arrays import *
    from trax.tf_numpy.numpy_impl.dtypes import *
    from trax.tf_numpy.numpy_impl.math_ops import *
    from trax.tf_numpy.numpy_impl.utils import finfo
    from trax.tf_numpy.numpy_impl.utils import promote_types
    from trax.tf_numpy.numpy_impl.utils import result_type
    # pylint: enable=wildcard-import

    max = amax  # pylint: disable=redefined-builtin,undefined-variable
    min = amin  # pylint: disable=redefined-builtin,undefined-variable
    round = around  # pylint: disable=redefined-builtin,undefined-variable

try:
  from tensorflow.python.ops.numpy_ops.np_config import enable_numpy_behavior
  # TODO(b/171429739): This should be moved to every individual file/test.
  enable_numpy_behavior()

except ImportError:
  pass

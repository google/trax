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

"""Tests for tf numpy random number methods."""
import numpy as np
import tensorflow.compat.v2 as tf
from six.moves import range

# Needed for ndarray.reshape.
from trax.tf_numpy.numpy_impl import random


class RandomTest(tf.test.TestCase):
    def assertNotAllClose(self, a, b, **kwargs):
        try:
            self.assertAllClose(a, b, **kwargs)
        except AssertionError:
            return
        raise AssertionError("The two values are close at all %d elements" % np.size(a))

    def testRandN(self):
        def run_test(*args):
            num_samples = 1000
            tol = 0.1  # High tolerance to keep the # of samples low else the test
            # takes a long time to run.
            random.seed(10)
            outputs = [random.randn(*args) for _ in range(num_samples)]

            # Test output shape.
            for output in outputs:
                self.assertEqual(output.shape, tuple(args))
                self.assertEqual(output.dtype.type, random.DEFAULT_RANDN_DTYPE)

            if np.prod(args):  # Don't bother with empty arrays.
                outputs = [output.tolist() for output in outputs]

                # Test that the properties of normal distribution are satisfied.
                mean = np.mean(outputs, axis=0)
                stddev = np.std(outputs, axis=0)
                self.assertAllClose(mean, np.zeros(args), atol=tol)
                self.assertAllClose(stddev, np.ones(args), atol=tol)

                # Test that outputs are different with different seeds.
                random.seed(20)
                diff_seed_outputs = [
                    random.randn(*args).tolist() for _ in range(num_samples)
                ]
                self.assertNotAllClose(outputs, diff_seed_outputs)

                # Test that outputs are the same with the same seed.
                random.seed(10)
                same_seed_outputs = [
                    random.randn(*args).tolist() for _ in range(num_samples)
                ]
                self.assertAllClose(outputs, same_seed_outputs)

        run_test()
        run_test(0)
        run_test(1)
        run_test(5)
        run_test(2, 3)
        run_test(0, 2, 3)
        run_test(2, 0, 3)
        run_test(2, 3, 0)
        run_test(2, 3, 5)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.test.main()

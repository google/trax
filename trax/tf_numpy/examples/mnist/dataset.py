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

"""Load pickled MNIST data."""
import gzip
import os
import pickle
import random
import urllib
import numpy as np


def load():
  """Loads the dataset.

  Looks for the dataset at /tmp/mnist.pkl.gz and downloads it if it is not there
  already.

  Note: The training data is shuffled.

  Returns:
    ((train_x, train_y), (valid_x, valid_y), (test_x, test_y)).
    Shapes:
      train_x: num_training_examples x image_size
      train_y: num_training_examples x num_classes
      valid_x: num_validation_examples x image_size
      valid_y: num_validation_examples x num_classes
      test_x: num_test_examples x image_size
      test_y: num_test_examples x num_classes
  """
  filepath = _maybe_download()
  with gzip.open(os.path.join(filepath), 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f)
  training_data = (training_data[0], [to_one_hot(x) for x in training_data[1]])
  validation_data = (validation_data[0],
                     [to_one_hot(x) for x in validation_data[1]])
  test_data = (test_data[0], [to_one_hot(x) for x in test_data[1]])

  def shuffle(data):
    zipped = zip(*data)
    random.shuffle(zipped)
    return zip(*zipped)

  return (shuffle(training_data), validation_data, test_data)


def to_one_hot(label, num_classes=10):
  vec = np.zeros(num_classes, dtype=np.float32)
  vec[label] = 1.
  return vec


def _maybe_download():
  """Downloads the MNIST dataset if it is not there already."""
  data_url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
  filename = data_url.split('/')[-1]
  filepath = os.path.join(_get_data_dir(), filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      print('\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))

    filepath, _ = urllib.urlretrieve(data_url, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Successfully downloaded %s %d bytes.' % (filename, statinfo.st_size))
  else:
    print('Data already present on disk.')
  return filepath


def _get_data_dir():
  return '/tmp'

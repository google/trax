# coding=utf-8
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

# Lint as: python3
"""Module for downloading model weights"""


import os
from pathlib import Path

import urllib

import zipfile

def find_ckpt_file_in_dir(dir_path):
  # todo(piotrekp1): get last checkpoint instead of any
  for fname in os.listdir(dir_path):
    if '.ckpt' in fname:
      return fname.split('.ckpt')[0] + '.ckpt'
  raise FileNotFoundError('Selected directory doesn\'t contain a ckpt file')

def cd_nested_directory(dir_path):
  while len(os.listdir(dir_path)) == 1:
    filename = os.listdir(dir_path)[0]
    new_path = os.path.join(dir_path, filename)
    if os.path.isdir(new_path):
      dir_path = new_path
  return dir_path


def download_weights_if_not_downloaded(link, model_name, download_dir=None):
  """
     if download dir contains any files (or is direct path to a model) then it returns this directory (path)
     default download_dir is ~/trax/models/${model_name}
     otherwise downloads model into download_dir and returns the directory path
  """
  if download_dir is None:
    # default directory is ~/trax/models/$MODEL_NAME
    download_dir = os.path.join('~', 'trax', 'models', model_name)
    download_dir = os.path.expanduser(download_dir)

  if os.path.exists(download_dir):
    if os.path.isdir(download_dir) and len(os.listdir(download_dir)) > 0:
      # model already exists, mdoel directory as input
      download_dir = cd_nested_directory(download_dir) # go in if nested single directories inside
      return download_dir, find_ckpt_file_in_dir(download_dir)
  with Path(download_dir) as p:
    p.mkdir(parents=True, exist_ok=True)

  if not link.endswith('.zip'):
    raise NotImplementedError(
          'Only downloading models packed with zip is implemented')

  # assumes model is packed with zip
  download_path = os.path.join(download_dir,  'model_temp')

  print(f'Downloading model from {link} to {download_dir}') # todo(piotrekp1) logging to stderr?
  urllib.request.urlretrieve(link, download_path)

  if not zipfile.is_zipfile(download_path):
    raise NotImplementedError(
          'Only downloading models packed with zip is implemented')

  # todo(piotrekp1): some locks to handle crashes during unpacking or download
  with zipfile.ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

  os.remove(download_path)
  download_dir = cd_nested_directory(download_dir)

  return download_dir, find_ckpt_file_in_dir(download_dir)

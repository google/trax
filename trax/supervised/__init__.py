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

"""Supervised learning imports in Trax."""

from trax.supervised import callbacks
from trax.supervised import decoding
from trax.supervised import lr_schedules
from trax.supervised import trainer_lib
from trax.supervised import training
from trax.supervised.trainer_lib import train
from trax.supervised.trainer_lib import Trainer
from trax.supervised.training import EvalTask
from trax.supervised.training import TrainTask

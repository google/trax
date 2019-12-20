## Trax &mdash; your path to advanced deep learning

![train tracks](https://images.pexels.com/photos/461772/pexels-photo-461772.jpeg?dl&fit=crop&crop=entropy&w=32&h=21)
[![PyPI
version](https://badge.fury.io/py/trax.svg)](https://badge.fury.io/py/trax)
[![GitHub
Issues](https://img.shields.io/github/issues/google/trax.svg)](https://github.com/google/trax/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

[Trax](https://github.com/google/trax) helps you understand deep learning.
We start with basic maths and go through
[layers](https://colab.research.google.com/github/google/trax/blob/master/trax/layers/intro.ipynb),
models, supervised and reinforcement learning.
We get to advanced deep learning results, including recent papers and
state-of-the-art models.

[Trax](https://github.com/google/trax) is a successor to the
[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library and is
actively used and maintained by researchers and engineers within the
[Google Brain team](https://research.google.com/teams/brain/) and a community
of users. We're eager to collaborate with you too, so feel free to
[open an issue on GitHub](https://github.com/google/trax/issues)
or send along a pull request (see [our contribution doc](CONTRIBUTING.md)).

### Examples

See our example layers in a TPU/GPU-backed colab notebook at
[Trax Demo](https://colab.research.google.com/github/google/trax/blob/master/trax/layers/intro.ipynb)

#### MLP on MNIST


```
python -m trax.trainer \
  --dataset=mnist \
  --model=MLP \
  --config="train.steps=1000"
```

#### Resnet50 on Imagenet


```
python -m trax.trainer \
  --config_file=$PWD/trax/configs/resnet50_imagenet_8gb.gin
```

#### TransformerDecoder on LM1B


```
python -m trax.trainer \
  --config_file=transformer_lm1b_8gb.gin
```


## Trax &mdash; your path to advanced deep learning

![train tracks](https://images.pexels.com/photos/461772/pexels-photo-461772.jpeg?dl&fit=crop&crop=entropy&w=32&h=21)
[![PyPI
version](https://badge.fury.io/py/trax.svg)](https://badge.fury.io/py/trax)
[![GitHub
Issues](https://img.shields.io/github/issues/google/trax.svg)](https://github.com/google/trax/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/trax-ml/community)


[Trax](https://github.com/google/trax) helps you understand and explore advanced deep learning.
We [focus](#structure) on making Trax code clear while pushing advanced models like
[Reformer](https://github.com/google/trax/tree/master/trax/models/reformer) to their limits.
Trax is actively used and maintained in the [Google Brain team](https://research.google.com/teams/brain/).
Give it a try, [talk to us](https://gitter.im/trax-ml/community)
or [open an issue](https://github.com/google/trax/issues) if needed.


### Use Trax

You can use Trax either as a library from your own python scripts and notebooks
or as a binary from the shell, which can be more convenient for training large models.
Trax includes a number of deep learning models (ResNet, Transformer, RNNs, ...)
and has bindings to a large number of deep learning datasets, including
[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [TensorFlow datasets](https://www.tensorflow.org/datasets/catalog/overview).
It runs without any changes on CPUs, GPUs and TPUs.

To see how to use Trax as a library, take a look at this [quick start colab](https://colab.research.google.com/github/google/trax/blob/master/trax/intro.ipynb)
which explains how to:

1. Create data in python.
1. Connect it to a Transformer model in Trax.
1. Train it and run inference.

With Colab, you can select a CPU or GPU runtime, or even get a free 8-core TPU as
runtime. Please note, with TPUs in colab you need to set extra flags as demonstrated in these
[training](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/text_generation.ipynb)
and [inference](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/image_generation.ipynb) colabs.

To use Trax as a binary, we recommend pairing your usage with [gin-config](https://github.com/google/gin-config)
to keep track of model type, learning rate, and hyper-parameters or training settings.

Take a look at [an example gin config](https://github.com/google/trax/blob/master/trax/configs/mlp_mnist.gin)
 for training a simple MLP on MNIST and run it as follows:

```
python -m trax.trainer --config_file=$PWD/trax/configs/mlp_mnist.gin
```

As a more advanced example, you can train a [Reformer](https://github.com/google/trax/tree/master/trax/models/reformer)
on [Imagenet64](https://arxiv.org/abs/1707.08819) to generate images [like this](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/image_generation.ipynb)
with the following command:

```
python -m trax.trainer --config_file=$PWD/trax/configs/reformer_imagenet64.gin
```

### Structure

Trax code is structured in a way that allows you to understand deep learning
from scratch. We start with basic maths and go through layers, models,
supervised and reinforcement learning. We get to advanced deep learning
results, including recent papers such as [Reformer - The Efficient Transformer](https://arxiv.org/abs/2001.04451),
selected for oral presentation at [ICLR 2020](https://iclr.cc/Conferences/2020/).

The main steps needed to understand deep learning correspond to sub-directories
in Trax code:

* [math/](https://github.com/google/trax/tree/master/trax/math) &mdash; basic math operations and ways to accelerate them on GPUs and TPUs (through [JAX](https://github.com/google/jax) and [TensorFlow](https://www.tensorflow.org/))
* [layers/](https://github.com/google/trax/tree/master/trax/layers) are the basic building blocks of neural networks and here you'll find how they are built and all the essentials
* [models/](https://github.com/google/trax/tree/master/trax/models) contains all basic models (MLP, ResNet, Transformer, ...) and a number of new research models
* [optimizers/](https://github.com/google/trax/tree/master/trax/optimizers) is a directory with optimizers needed for deep learning
* [supervised/](https://github.com/google/trax/tree/master/trax/supervised) contains the utilities needed to run supervised learning and the Trainer class
* [rl/](https://github.com/google/trax/tree/master/trax/rl) contains our work on reinforcement learning

### Development

To get the most recent update on Trax development, [chat with us](https://gitter.im/trax-ml/community).

Most common supervised learning models in Trax are running and should have clear
code &mdash; if this is not the case, please [open an issue](https://github.com/google/trax/issues)
or, even better, send along a pull request (see [our contribution doc](CONTRIBUTING.md)).
In Trax we value documentation, examples and colabs so if you find any
problems with those, please report it and contribute a solution.

We are still improving a few smaller parts of [layers](https://github.com/google/trax/tree/master/trax/layers),
planning to update the [supervised](https://github.com/google/trax/tree/master/trax/supervised) API and
heavily working on the [rl](https://github.com/google/trax/tree/master/trax/rl) part,
so expect these parts to change over the next few months. We are also working hard
to improve our documentation and examples and we welcome help with that.

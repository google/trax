# Trax &mdash; Deep Learning with Clear Code and Speed

![train tracks](https://images.pexels.com/photos/461772/pexels-photo-461772.jpeg?dl&fit=crop&crop=entropy&w=32&h=21)
[![PyPI
version](https://badge.fury.io/py/trax.svg)](https://badge.fury.io/py/trax)
[![GitHub
Issues](https://img.shields.io/github/issues/google/trax.svg)](https://github.com/google/trax/issues)
![GitHub Build](https://github.com/google/trax/actions/workflows/build.yaml/badge.svg)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/trax-ml/community)

[Trax](https://trax-ml.readthedocs.io/en/latest/) is an end-to-end library for deep learning that focuses on clear code and speed. It is actively used and maintained in the [Google Brain team](https://research.google.com/teams/brain/). This notebook ([run it in colab](https://colab.research.google.com/github/google/trax/blob/master/trax/intro.ipynb)) shows how to use Trax and where you can find more information.

  1. **Run a pre-trained Transformer**: create a translator in a few lines of code
  1. **Features and resources**: [API docs](https://trax-ml.readthedocs.io/en/latest/), where to [talk to us](https://gitter.im/trax-ml/community), how to [open an issue](https://github.com/google/trax/issues) and more
  1. **Walkthrough**: how Trax works, how to make new models and train on your own data

We welcome **contributions** to Trax! We welcome PRs with code for new models and layers as well as improvements to our code and documentation. We especially love **notebooks** that explain how models work and show how to use them to solve problems!



Here are a few example notebooks:-

* [**trax.data API explained**](https://github.com/google/trax/blob/master/trax/examples/trax_data_Explained.ipynb) : Explains some of the major functions in the `trax.data` API 
* [**Named Entity Recognition using Reformer**](https://github.com/google/trax/blob/master/trax/examples/NER_using_Reformer.ipynb) : Uses a [Kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) for implementing Named Entity Recognition using the [Reformer](https://arxiv.org/abs/2001.04451) architecture.
* [**Deep N-Gram models**](https://github.com/google/trax/blob/master/trax/examples/Deep_N_Gram_Models.ipynb) : Implementation of deep n-gram models trained on Shakespeares works



**General Setup**

Execute the following cell (once) before running any of the code samples.


```python
import os
import numpy as np

!pip install -q -U trax
import trax
```


## 1. Run a pre-trained Transformer

Here is how you create an English-German translator in a few lines of code:

* create a Transformer model in Trax with [trax.models.Transformer](https://trax-ml.readthedocs.io/en/latest/trax.models.html#trax.models.transformer.Transformer)
* initialize it from a file with pre-trained weights with [model.init_from_file](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.base.Layer.init_from_file)
* tokenize your input sentence to input into the model with [trax.data.tokenize](https://trax-ml.readthedocs.io/en/latest/trax.data.html#trax.data.tf_inputs.tokenize)
* decode from the Transformer with [trax.supervised.decoding.autoregressive_sample](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.decoding.autoregressive_sample)
* de-tokenize the decoded result to get the translation with [trax.data.detokenize](https://trax-ml.readthedocs.io/en/latest/trax.data.html#trax.data.tf_inputs.detokenize)



```python
# Create a Transformer model.
# Pre-trained model config in gs://trax-ml/models/translation/ende_wmt32k.gin
model = trax.models.Transformer(
    input_vocab_size=33300,
    d_model=512, d_ff=2048,
    n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
    max_len=2048, mode='predict')

# Initialize using pre-trained weights.
model.init_from_file('gs://trax-ml/models/translation/ende_wmt32k.pkl.gz',
                     weights_only=True)

# Tokenize a sentence.
sentence = 'It is nice to learn new things today!'
tokenized = list(trax.data.tokenize(iter([sentence]),  # Operates on streams.
                                    vocab_dir='gs://trax-ml/vocabs/',
                                    vocab_file='ende_32k.subword'))[0]

# Decode from the Transformer.
tokenized = tokenized[None, :]  # Add batch dimension.
tokenized_translation = trax.supervised.decoding.autoregressive_sample(
    model, tokenized, temperature=0.0)  # Higher temperature: more diverse results.

# De-tokenize,
tokenized_translation = tokenized_translation[0][:-1]  # Remove batch and EOS.
translation = trax.data.detokenize(tokenized_translation,
                                   vocab_dir='gs://trax-ml/vocabs/',
                                   vocab_file='ende_32k.subword')
print(translation)
```

    Es ist schön, heute neue Dinge zu lernen!


## 2. Features and resources

Trax includes basic models (like [ResNet](https://github.com/google/trax/blob/master/trax/models/resnet.py#L70), [LSTM](https://github.com/google/trax/blob/master/trax/models/rnn.py#L100), [Transformer](https://github.com/google/trax/blob/master/trax/models/transformer.py#L189)) and RL algorithms
(like [REINFORCE](https://github.com/google/trax/blob/master/trax/rl/training.py#L244), [A2C](https://github.com/google/trax/blob/master/trax/rl/actor_critic_joint.py#L458), [PPO](https://github.com/google/trax/blob/master/trax/rl/actor_critic_joint.py#L209)). It is also actively used for research and includes
new models like the [Reformer](https://github.com/google/trax/tree/master/trax/models/reformer) and new RL algorithms like [AWR](https://arxiv.org/abs/1910.00177). Trax has bindings to a large number of deep learning datasets, including
[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [TensorFlow datasets](https://www.tensorflow.org/datasets/catalog/overview).


You can use Trax either as a library from your own python scripts and notebooks
or as a binary from the shell, which can be more convenient for training large models.
It runs without any changes on CPUs, GPUs and TPUs.

* [API docs](https://trax-ml.readthedocs.io/en/latest/)
* [chat with us](https://gitter.im/trax-ml/community)
* [open an issue](https://github.com/google/trax/issues)
* subscribe to [trax-discuss](https://groups.google.com/u/1/g/trax-discuss) for news


## 3. Walkthrough

You can learn here how Trax works, how to create new models and how to train them on your own data.

### Tensors and Fast Math

The basic units flowing through Trax models are *tensors* - multi-dimensional arrays, sometimes also known as numpy arrays, due to the most widely used package for tensor operations -- `numpy`. You should take a look at the [numpy guide](https://numpy.org/doc/stable/user/quickstart.html) if you don't know how to operate on tensors: Trax also uses the numpy API for that.

In Trax we want numpy operations to run very fast, making use of GPUs and TPUs to accelerate them. We also want to automatically compute gradients of functions on tensors. This is done in the `trax.fastmath` package thanks to its backends -- [JAX](https://github.com/google/jax) and [TensorFlow numpy](https://tensorflow.org/guide/tf_numpy).


```python
from trax.fastmath import numpy as fastnp
trax.fastmath.use_backend('jax')  # Can be 'jax' or 'tensorflow-numpy'.

matrix  = fastnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f'matrix = \n{matrix}')
vector = fastnp.ones(3)
print(f'vector = {vector}')
product = fastnp.dot(vector, matrix)
print(f'product = {product}')
tanh = fastnp.tanh(product)
print(f'tanh(product) = {tanh}')
```

    matrix = 
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    vector = [1. 1. 1.]
    product = [12. 15. 18.]
    tanh(product) = [0.99999994 0.99999994 0.99999994]


Gradients can be calculated using `trax.fastmath.grad`.


```python
def f(x):
  return 2.0 * x * x

grad_f = trax.fastmath.grad(f)

print(f'grad(2x^2) at 1 = {grad_f(1.0)}')
```

    grad(2x^2) at 1 = 4.0


### Layers

Layers are basic building blocks of Trax models. You will learn all about them in the [layers intro](https://trax-ml.readthedocs.io/en/latest/notebooks/layers_intro.html) but for now, just take a look at the implementation of one core Trax layer, `Embedding`:

```python
class Embedding(base.Layer):
  """Trainable layer that maps discrete tokens/IDs to vectors."""

  def __init__(self,
               vocab_size,
               d_feature,
               kernel_initializer=init.RandomNormalInitializer(1.0)):
    """Returns an embedding layer with given vocabulary size and vector size.

    Args:
      vocab_size: Size of the input vocabulary. The layer will assign a unique
          vector to each ID in `range(vocab_size)`.
      d_feature: Dimensionality/depth of the output vectors.
      kernel_initializer: Function that creates (random) initial vectors for
          the embedding.
    """
    super().__init__(name=f'Embedding_{vocab_size}_{d_feature}')
    self._d_feature = d_feature  # feature dimensionality
    self._vocab_size = vocab_size
    self._kernel_initializer = kernel_initializer

  def forward(self, x):
    """Returns embedding vectors corresponding to input token IDs.

    Args:
      x: Tensor of token IDs.

    Returns:
      Tensor of embedding vectors.
    """
    return jnp.take(self.weights, x, axis=0, mode='clip')

  def init_weights_and_state(self, input_signature):
    """Returns tensor of newly initialized embedding vectors."""
    del input_signature
    shape_w = (self._vocab_size, self._d_feature)
    w = self._kernel_initializer(shape_w, self.rng)
    self.weights = w
```

Layers with trainable weights like `Embedding` need to be initialized with the signature (shape and dtype) of the input, and then can be run by calling them.



```python
from trax import layers as tl

# Create an input tensor x.
x = np.arange(15)
print(f'x = {x}')

# Create the embedding layer.
embedding = tl.Embedding(vocab_size=20, d_feature=32)
embedding.init(trax.shapes.signature(x))

# Run the layer -- y = embedding(x).
y = embedding(x)
print(f'shape of y = {y.shape}')
```

    x = [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
    shape of y = (15, 32)


### Models

Models in Trax are built from layers most often using the `Serial` and `Branch` combinators. You can read more about those combinators in the [layers intro](https://trax-ml.readthedocs.io/en/latest/notebooks/layers_intro.html) and
see the code for many models in `trax/models/`, e.g., this is how the [Transformer Language Model](https://github.com/google/trax/blob/master/trax/models/transformer.py#L167) is implemented. Below is an example of how to build a sentiment classification model.


```python
model = tl.Serial(
    tl.Embedding(vocab_size=8192, d_feature=256),
    tl.Mean(axis=1),  # Average on axis 1 (length of sentence).
    tl.Dense(2),      # Classify 2 classes.
    tl.LogSoftmax()   # Produce log-probabilities.
)

# You can print model structure.
print(model)
```

    Serial[
      Embedding_8192_256
      Mean
      Dense_2
      LogSoftmax
    ]


### Data

To train your model, you need data. In Trax, data streams are represented as python iterators, so you can call `next(data_stream)` and get a tuple, e.g., `(inputs, targets)`. Trax allows you to use [TensorFlow Datasets](https://www.tensorflow.org/datasets) easily and you can also get an iterator from your own text file using the standard `open('my_file.txt')`.


```python
train_stream = trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True)()
eval_stream = trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=False)()
print(next(train_stream))  # See one example.
```

    (b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it.", 0)


Using the `trax.data` module you can create input processing pipelines, e.g., to tokenize and shuffle your data. You create data pipelines using `trax.data.Serial` and they are functions that you apply to streams to create processed streams.


```python
data_pipeline = trax.data.Serial(
    trax.data.Tokenize(vocab_file='en_8k.subword', keys=[0]),
    trax.data.Shuffle(),
    trax.data.FilterByLength(max_length=2048, length_keys=[0]),
    trax.data.BucketByLength(boundaries=[  32, 128, 512, 2048],
                             batch_sizes=[256,  64,  16,    4, 1],
                             length_keys=[0]),
    trax.data.AddLossWeights()
  )
train_batches_stream = data_pipeline(train_stream)
eval_batches_stream = data_pipeline(eval_stream)
example_batch = next(train_batches_stream)
print(f'shapes = {[x.shape for x in example_batch]}')  # Check the shapes.
```

    shapes = [(4, 1024), (4,), (4,)]


### Supervised training

When you have the model and the data, use `trax.supervised.training` to define training and eval tasks and create a training loop. The Trax training loop optimizes training and will create TensorBoard logs and model checkpoints for you.


```python
from trax.supervised import training

# Training task.
train_task = training.TrainTask(
    labeled_data=train_batches_stream,
    loss_layer=tl.WeightedCategoryCrossEntropy(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=500,
)

# Evaluaton task.
eval_task = training.EvalTask(
    labeled_data=eval_batches_stream,
    metrics=[tl.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy()],
    n_eval_batches=20  # For less variance in eval numbers.
)

# Training loop saves checkpoints to output_dir.
output_dir = os.path.expanduser('~/output_dir/')
!rm -rf {output_dir}
training_loop = training.Loop(model,
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)

# Run 2000 steps (batches).
training_loop.run(2000)
```

    
    Step      1: Ran 1 train steps in 0.78 secs
    Step      1: train WeightedCategoryCrossEntropy |  1.33800304
    Step      1: eval  WeightedCategoryCrossEntropy |  0.71843582
    Step      1: eval      WeightedCategoryAccuracy |  0.56562500
    
    Step    500: Ran 499 train steps in 5.77 secs
    Step    500: train WeightedCategoryCrossEntropy |  0.62914723
    Step    500: eval  WeightedCategoryCrossEntropy |  0.49253047
    Step    500: eval      WeightedCategoryAccuracy |  0.74062500
    
    Step   1000: Ran 500 train steps in 5.03 secs
    Step   1000: train WeightedCategoryCrossEntropy |  0.42949259
    Step   1000: eval  WeightedCategoryCrossEntropy |  0.35451687
    Step   1000: eval      WeightedCategoryAccuracy |  0.83750000
    
    Step   1500: Ran 500 train steps in 4.80 secs
    Step   1500: train WeightedCategoryCrossEntropy |  0.41843575
    Step   1500: eval  WeightedCategoryCrossEntropy |  0.35207348
    Step   1500: eval      WeightedCategoryAccuracy |  0.82109375
    
    Step   2000: Ran 500 train steps in 5.35 secs
    Step   2000: train WeightedCategoryCrossEntropy |  0.38129005
    Step   2000: eval  WeightedCategoryCrossEntropy |  0.33760912
    Step   2000: eval      WeightedCategoryAccuracy |  0.85312500


After training the model, run it like any layer to get results.


```python
example_input = next(eval_batches_stream)[0][0]
example_input_str = trax.data.detokenize(example_input, vocab_file='en_8k.subword')
print(f'example input_str: {example_input_str}')
sentiment_log_probs = model(example_input[None, :])  # Add batch dimension.
print(f'Model returned sentiment probabilities: {np.exp(sentiment_log_probs)}')
```

    example input_str: I first saw this when I was a teen in my last year of Junior High. I was riveted to it! I loved the special effects, the fantastic places and the trial-aspect and flashback method of telling the story.<br /><br />Several years later I read the book and while it was interesting and I could definitely see what Swift was trying to say, I think that while it's not as perfect as the book for social commentary, as a story the movie is better. It makes more sense to have it be one long adventure than having Gulliver return after each voyage and making a profit by selling the tiny Lilliput sheep or whatever.<br /><br />It's much more arresting when everyone thinks he's crazy and the sheep DO make a cameo anyway. As a side note, when I saw Laputa I was stunned. It looks very much like the Kingdom of Zeal from the Chrono Trigger video game (1995) that also made me like this mini-series even more.<br /><br />I saw it again about 4 years ago, and realized that I still enjoyed it just as much. Really high quality stuff and began an excellent run of Sweeps mini-series for NBC who followed it up with the solid Merlin and interesting Alice in Wonderland.<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    Model returned sentiment probabilities: [[3.984500e-04 9.996014e-01]]

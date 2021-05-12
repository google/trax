## Reformer and Terraformer: The Efficient Transformers

Reformer and Terraformer are more efficient versions of Transformer that uses reversible layers, locality-sensitive hashing and sparse layers.

### Papers

* Reformer: Read about the details of Reformer in the [Reformer paper](https://arxiv.org/abs/2001.04451) which was selected for oral presentation at [ICLR 2020](https://iclr.cc/Conferences/2020/).


* Terraformer: Read about the details of Terraformer in the following paper.

### Models


* Generate images with Reformer using [this colab](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/image_generation.ipynb).

* Translate from English to German with a reversible encoder-decoder model using [this colab](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/machine_translation.ipynb).

* Medium-size (~700M weights) Terraformer model for summarizing arxiv articles is available at `gs://trax-ml/terraformer/medium`

* Large (~7B weights) Terraformer model pre-trained on C4 `gs://trax-ml/terraformer/big`

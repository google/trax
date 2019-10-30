## `Trax`

![train tracks](https://images.pexels.com/photos/461772/pexels-photo-461772.jpeg?dl&fit=crop&crop=entropy&w=640&h=426)

### Examples

#### Example Colab

See our example constructing language models from scratch in a GPU-backed colab notebook at
[Trax Demo](https://colab.research.google.com/github/google/trax/blob/master/trax/notebooks/trax_demo_iclr2019.ipynb)

#### MLP on MNIST


```
python -m trax.trainer \
  --dataset=mnist \
  --model=MLP \
  --config="train.train_steps=1000"
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


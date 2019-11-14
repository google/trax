# Trax Layers



## Base layer structure

All layers inherit from the Layer class and generally need to implement 2
methods:

```python
def forward(self, inputs, weights):
  """Computes the layer's output as part of a forward pass through the model."""

def new_weights(self, input_signature):
    """Returns new weights suitable for inputs with the given signature.
```

The base Layer class wraps these functions and provides initialization
and call functions to be used as follows.

```python
layer = MyLayer()
x = np.zeros(10)
layer.initialize_once(signature(x))
output = layer(x)
```

## Decorator

To create simple layers, especially ones without parameters, use the layer
decorator.

```python
@base.layer()
def Relu(x, **unused_kwargs):
  return np.maximum(x, 0.)
```

## Parameter sharing

Parameters are shared when the same layer object is used.

```python
standard_mlp = layers.Serial(layers.Dense(10), layers.Dense(10))
layer = Dense(10)
shared_parameters_mlp = layers.Serial(layer, layer)
```
For this reason, if you call `layer.initialize_once(...)` for the second time
on an already initialized layer, it will not re-initialize the layer.

## Core layers

* Dense
* Conv

## Layer composition

* Serial
* Parallel

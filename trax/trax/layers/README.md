# Trax Layers



## Base layer structure

All layers inherit from the Layer class and generally need to implement 2
methods:

```python
def forward(self, inputs):
  """Computes the layer's output as part of a forward pass through the model."""

def init_weights_and_state(self, input_signature):
    """Initializes weights and state for inputs with the given signature."""
```

The base Layer class wraps these functions and provides initialization
and call functions to be used as follows.

```python
layer = MyLayer()
x = np.zeros(10)
layer.init(signature(x))
output = layer(x)
```

## Fn layer

To create simple layers without parameters, use the Fn layer.

```python
def Relu(x):
  return Fn('Relu', lambda x: np.maximum(x, np.zeros_like(x)))
```

## Parameter sharing

Parameters are shared when the same layer object is used.

```python
standard_mlp = layers.Serial(layers.Dense(10), layers.Dense(10))
layer = Dense(10)
shared_parameters_mlp = layers.Serial(layer, layer)
```

## Core layers

* Dense
* Conv

## Layer composition

* Serial
* Parallel

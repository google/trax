### Current TensorFlow Numpy API (in trax)

(The parenthesis contains the according numpy equivalence)

1. `trax.tf_numpy.numpy.array_ops`
    * `trax.tf_numpy.numpy.array_ops.array` ([`np.array`](https://numpy.org/doc/1.18/reference/generated/numpy.array.html))
    * `trax.tf_numpy.numpy.array_ops.asarray` ([`np.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html))
    * `trax.tf_numpy.numpy.array_ops.asanyarray` ([`np.asanyarray`](https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html))
    * `trax.tf_numpy.numpy.array_ops.ascontiguousarray` ([`np.ascontiguousarray`](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html))
    * `trax.tf_numpy.numpy.array_ops.arange` ([`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html))
    * `trax.tf_numpy.numpy.array_ops.all` ([`np.all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html))
    * `trax.tf_numpy.numpy.array_ops.any` ([`np.any`](https://numpy.org/doc/1.18/reference/generated/numpy.any.html))
    * `trax.tf_numpy.numpy.array_ops.around` ([`np.around`](https://numpy.org/doc/1.18/reference/generated/numpy.around.html))
    * `trax.tf_numpy.numpy.array_ops.amax` ([`np.amax`](https://docs.scipy.org/doc/numpy-1.9.3/reference/generated/numpy.amax.html))
    * `trax.tf_numpy.numpy.array_ops.amin` ([`np.amin`](https://numpy.org/doc/1.18/reference/generated/numpy.amin.html))
    * `trax.tf_numpy.numpy.array_ops.broadcast_to` ([`np.broadcast_to`](https://numpy.org/doc/1.18/reference/generated/numpy.broadcast_to.html))  
    * `trax.tf_numpy.numpy.array_ops.compress` ([`np.compress`](https://numpy.org/doc/1.18/reference/generated/numpy.compress.html))
    * `trax.tf_numpy.numpy.array_ops.copy` ([`np.copy`](https://numpy.org/doc/1.18/reference/generated/numpy.copy.html))
    * `trax.tf_numpy.numpy.array_ops.cumprod` ([`np.cumprod`](https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html))
    * `trax.tf_numpy.numpy.array_ops.cumsum` ([`np.cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html))
    * `trax.tf_numpy.numpy.array_ops.diag` ([`np.diag`](https://numpy.org/doc/stable/reference/generated/numpy.diag.html))
    * `trax.tf_numpy.numpy.array_ops.diagflat` ([`np.diagflat`](https://numpy.org/doc/1.18/reference/generated/numpy.diagflat.html))
    * `trax.tf_numpy.numpy.array_ops.expand_dims` ([`np.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html))
    * `trax.tf_numpy.numpy.array_ops.empty` ([`np.empty`](https://numpy.org/doc/1.18/reference/generated/numpy.empty.html))
    * `trax.tf_numpy.numpy.array_ops.empty_like` ([`np.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html))
    * `trax.tf_numpy.numpy.array_ops.eye` ([`np.eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html))
    * `trax.tf_numpy.numpy.array_ops.full` ([`np.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html))
    * `trax.tf_numpy.numpy.array_ops.full_like` ([`np.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html))
    * `trax.tf_numpy.numpy.array_ops.geomspace` ([`np.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html))
    * `trax.tf_numpy.numpy.array_ops.imag` ([`np.imag`](https://numpy.org/doc/stable/reference/generated/numpy.imag.html))
    * `trax.tf_numpy.numpy.array_ops.isscalar`
    * `trax.tf_numpy.numpy.array_ops.identity` ([`np.identity`](https://numpy.org/doc/stable/reference/generated/numpy.identity.html))
    * `trax.tf_numpy.numpy.array_ops.moveaxis` ([`np.moveaxis`](https://numpy.org/doc/1.18/reference/generated/numpy.moveaxis.html))
    * `trax.tf_numpy.numpy.array_ops.mean` ([`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html))
    * `trax.tf_numpy.numpy.array_ops.ndim`
    * `trax.tf_numpy.numpy.array_ops.ones` ([`np.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html))
    * `trax.tf_numpy.numpy.array_ops.ones_like` ([`np.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html))
    * `trax.tf_numpy.numpy.array_ops.pad`
    * `trax.tf_numpy.numpy.array_ops.prod` ([`np.prod`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.prod.html))
    * `trax.tf_numpy.numpy.array_ops.ravel` ([`np.ravel`](https://numpy.org/doc/1.18/reference/generated/numpy.ravel.html))
    * `trax.tf_numpy.numpy.array_ops.real` ([`np.real`](https://numpy.org/doc/1.18/reference/generated/numpy.real.html))
    * `trax.tf_numpy.numpy.array_ops.repeat` ([`np.repeat`](https://numpy.org/doc/1.18/reference/generated/numpy.repeat.html))
    * `trax.tf_numpy.numpy.array_ops.reshape` ([`np.reshape`](https://numpy.org/doc/1.18/reference/generated/numpy.reshape.html))
    * `trax.tf_numpy.numpy.array_ops.swapaxes`
    * `trax.tf_numpy.numpy.array_ops.split`
    * `trax.tf_numpy.numpy.array_ops.squeeze` ([`np.squeeze`](https://numpy.org/doc/1.18/reference/generated/numpy.squeeze.html))
    * `trax.tf_numpy.numpy.array_ops.sum` ([`np.sum`](https://numpy.org/doc/1.18/reference/generated/numpy.sum.html))
    * `trax.tf_numpy.numpy.array_ops.transpose` ([`np.transpose`](https://numpy.org/doc/1.18/reference/generated/numpy.transpose.html))
    * `trax.tf_numpy.numpy.array_ops.take` ([`np.take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html))
    * `trax.tf_numpy.numpy.array_ops.where`
    * `trax.tf_numpy.numpy.array_ops.zeros` ([`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html))
    * `trax.tf_numpy.numpy.array_ops.zeros_like` ([`np.zeros_like`](https://numpy.org/doc/1.18/reference/generated/numpy.zeros_like.html))

2. `trax.tf_numpy.numpy.arrays`
    * `trax.tf_numpy.numpy.arrays.ndarray`

3. `trax.tf_numpy.numpy.math_ops`
    * `trax.tf_numpy.numpy.math_ops.average`
    * `trax.tf_numpy.numpy.math_ops.argsort` ([`np.argsort`](https://numpy.org/devdocs/reference/generated/numpy.argsort.html))
    * `trax.tf_numpy.numpy.math_ops.argmax` ([`np.argmin`](https://numpy.org/devdocs/reference/generated/numpy.argmin.html))
    * `trax.tf_numpy.numpy.math_ops.clip` ([`np.clip`](https://numpy.org/doc/1.18/reference/generated/numpy.clip.html))
    * `trax.tf_numpy.numpy.math_ops.dot` ([`np.dot`](https://numpy.org/devdocs/reference/generated/numpy.dot.html))
    * `trax.tf_numpy.numpy.math_ops.exp` ([`np.exp`](https://numpy.org/devdocs/reference/generated/numpy.exp.html))
    * `trax.tf_numpy.numpy.math_ops.isclose` ([`np.isclose`](https://numpy.org/doc/1.18/reference/generated/numpy.isclose.html))
    * `trax.tf_numpy.numpy.math_ops.log` ([`np.log`](https://numpy.org/doc/1.18/reference/generated/numpy.log.html))
    * `trax.tf_numpy.numpy.math_ops.linspace` ([`np.linspace`](https://numpy.org/devdocs/reference/generated/numpy.linspace.html))
    * `trax.tf_numpy.numpy.math_ops.logspace` ([`np.logspace`](https://numpy.org/doc/1.18/reference/generated/numpy.logspace.html))
    * `trax.tf_numpy.numpy.math_ops.minimum` ([`np.minimum`](https://numpy.org/doc/1.18/reference/generated/numpy.minimum.html))
    * `trax.tf_numpy.numpy.math_ops.maximum` ([`np.maximum`](https://numpy.org/devdocs/reference/generated/numpy.maximum.html))
    * `trax.tf_numpy.numpy.math_ops.matmul` ([`np.matmul`](https://numpy.org/devdocs/reference/generated/numpy.matmul.html))
    * `trax.tf_numpy.numpy.math_ops.ptp` ([`np.ptp`](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html))
    * `trax.tf_numpy.numpy.math_ops.sqrt` ([`np.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html))
    * `trax.tf_numpy.numpy.math_ops.tanh` ([`np.tanh`](https://numpy.org/doc/stable/reference/generated/numpy.tanh.html)
    )


4. `trax.tf_numpy.numpy.random`
    * `trax.tf_numpy.numpy.random.seed`
    * `trax.tf_numpy.numpy.random.randn`
    * `trax.tf_numpy.numpy.random.DEFAULT_RANDN_DTYPE`

5. `trax.tf_numpy.numpy.utils`
    * `trax.tf_numpy.numpy.utils.np_doc`

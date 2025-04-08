import time

import numpy as np

from absl import logging
from layers import CrossEntropyLossWithLogSoftmax

import trax.fastmath as fastmath

from trax import layers as tl
from trax import optimizers, shapes

# from trax.data.encoder import encoder as encoder
# from trax.data.loader.tf import base as dataset
# from trax.data.preprocessing import inputs as preprocessing
from trax.fastmath import numpy as jnp


def Transpose():  # pylint: disable=invalid-name
    layer_name = (
        "Transpose"  # don't forget to give your custom layer a name to identify
    )

    # Custom function for the custom layer
    def f(x):  # pylint: disable=invalid-name
        assert len(x.shape) == 3 or len(x.shape) == 2, (
            "Houston we've got a problem: "
            "Cannot automatically reshape this "
            "stream - input is not a 2d or 3d array "
            "you should use trax.data.Batch(n) firsts, where n >= 1."
        )
        if len(x.shape) == 2:
            return jnp.transpose(x)

        return jnp.transpose(x, (0, 2, 1))

    return tl.Fn(layer_name, f, n_out=1)


def run_training(device_type="cpu", num_steps=100):
    """Run training with specified device configuration"""

    # ====== Determine the target device =======
    if device_type == "cpu":
        target_device = fastmath.devices("cpu")[0]
    elif device_type == "gpu":
        target_device = fastmath.devices("gpu")[0]
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")

    # Set the logging level to INFO or lower
    logging.set_verbosity(logging.INFO)

    print(f"\n\n{'='*20} RUNNING ON {device_type.upper()} {'='*20}")
    logging.info(f"Backend name: {fastmath.backend_name()}")
    logging.info(f"Backend device count: {fastmath.global_device_count()}")
    logging.info(f"Backend local device count: {fastmath.local_device_count()}")
    logging.info(f"JAX devices: {fastmath.devices(device_type)[0]}")
    logging.info(f"JAX target device: {fastmath.devices(device_type)[0]}")


    # ====== Create data pipeline =======
    # VOCAB_TYPE = "subword"
    # VOCAB_FILE = "en_8k.subword"
    #
    # vocab_size = encoder.vocab_size(VOCAB_TYPE, VOCAB_FILE)
    #
    # train_stream = dataset.TFDS('imdb_reviews', keys=('text', 'label'), train=True)()
    # eval_stream = dataset.TFDS('imdb_reviews', keys=('text', 'label'), train=False)()
    #
    # data_pipeline = preprocessing.Serial(
    #     preprocessing.ConvertToUnicode(keys=[0]),
    #     encoder.Tokenize(keys=[0], vocab_type=VOCAB_TYPE, vocab_file=VOCAB_FILE),
    #     preprocessing.Shuffle(),
    #     #preprocessing.FilterByLength(max_length=1000_000, length_keys=[0]),
    #     preprocessing.AddLossWeights(),
    #     lambda g: map(lambda x: (x[0], np.asarray(x[1]), x[2]), g),
    #     preprocessing.ClassificationVector(vocab_size=vocab_size),
    #     preprocessing.Batch(batch_size=32),
    #     lambda g: map(lambda x: (jnp.asarray(x[0]), jnp.asarray(x[1]), jnp.asarray(x[2])), g),
    # )

    def create_batch_generator(
        batch_size=32, feature_dim=10_000, num_classes=20, seed=42
    ):
        """
        Creates a generator that yields random example batches.

        Args:
            batch_size: Size of each batch
            feature_dim: Dimension of feature vectors
            num_classes: Number of possible classes
            seed: Random seed for reproducibility

        Returns:
            A generator that yields (features, labels, weights) tuples
        """
        # Initialize the RNG key
        key = fastmath.random.get_prng(seed)

        while True:
            # Split the key for this iteration to get two independent random keys
            key, subkey1, subkey2 = fastmath.random.split(key, 3)

            # Generate features, labels and weights
            features = fastmath.random.randint(
                subkey1, (batch_size, feature_dim), minval=0, maxval=10_000
            )
            labels = fastmath.random.randint(
                subkey2, (batch_size,), minval=0, maxval=num_classes
            )
            weights = jnp.ones((batch_size,))

            # Yield the batch
            yield (features, labels, weights)


    train_batches_stream = create_batch_generator()
    example_batch = next(train_batches_stream)  # Cache first batch to ensure fair comparison
    # train_batches_stream = data_pipeline(train_stream)
    # example_batch = next(train_batches_stream)  # Cache first batch to ensure fair comparison

    # ====== Create and initialize model =======
    mode = "train"
    model = tl.Serial(
        tl.Embedding(vocab_size=10_000, d_feature=1),
        Transpose(),
        tl.Dropout(rate=0.1, mode=mode),
        tl.LeakyRelu(a=0.1),
        tl.Dense(2, use_bias=False),
    )

    model_with_loss = tl.Serial(model, CrossEntropyLossWithLogSoftmax())

    # Initialize model
    print("Initializing model...")
    init_start = time.time()
    _, _ = model_with_loss.init(shapes.signature(example_batch))
    init_time = time.time() - init_start
    print(f"Model initialization time: {init_time:.4f} seconds")

    # First run to compile
    print("Compiling model with first batch...")
    compile_start = time.time()
    y = model_with_loss(example_batch)
    compile_time = time.time() - compile_start
    print(f"Compilation time: {compile_time:.4f} seconds")

    # Setup optimizer
    #optimizer = optimizers.Adafactor(0.001)
    optimizer = optimizers.SGD(0.0001)
    trainer = optimizers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)

    # ====== Training loop with timing =======
    print(f"Starting training for {num_steps} steps...")
    training_start = time.time()
    losses = []

    with fastmath.jax.jax.default_device(target_device):
        for i in range(num_steps):
            step_start = time.time()

            # Split the RNG to get a new key for this step
            step_rng, base_rng = fastmath.random.split(base_rng)

            # Get batch (use cached first batch for first iteration to ensure fair comparison)
            if i == 0:
                batch = example_batch
            else:
                batch = next(train_batches_stream)

            # Training step
            loss = trainer.one_step(batch, step_rng, step=i)
            step_time = time.time() - step_start
            losses.append(loss)

            # Print progress
            if i % 10 == 0 or i == num_steps - 1:
                print(f"Step {i}, Loss: {loss:.4f}, Step time: {step_time:.4f} seconds")

    training_time = time.time() - training_start
    avg_step_time = training_time / num_steps

    print(f"\n{'='*50}")
    print(f"Device: {device_type.upper()}")
    print(f"Total training time for {num_steps} steps: {training_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.4f} seconds")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"{'='*50}\n")

    return {
        "device": device_type,
        "init_time": init_time,
        "compile_time": compile_time,
        "total_training_time": training_time,
        "avg_step_time": avg_step_time,
        "final_loss": losses[-1]
    }

# Run and compare
NUM_STEPS = 5_000  # Use a smaller number for testing, then increase for full benchmark

# CPU run
cpu_results = run_training(device_type="cpu", num_steps=NUM_STEPS)

# GPU run
gpu_results = run_training(device_type="gpu", num_steps=NUM_STEPS)

# Print comparison
print("\n" + "="*50)
print("PERFORMANCE COMPARISON: CPU vs GPU")
print("="*50)
print(f"{'Metric':<25} {'CPU':<15} {'GPU':<15} {'Speedup':<10}")
print("-"*65)

for metric in ["init_time", "compile_time", "total_training_time", "avg_step_time"]:
    cpu_val = cpu_results[metric]
    gpu_val = gpu_results[metric]
    speedup = cpu_val / gpu_val if gpu_val > 0 else float('inf')
    print(f"{metric:<25} {cpu_val:.4f}s{' ':<9} {gpu_val:.4f}s{' ':<9} {speedup:.2f}x")

print("="*65)


def create_batch_generator(batch_size=32, feature_dim=10_000, num_classes=20, seed=42):
    """
    Creates a generator that yields random example batches.

    Args:
        batch_size: Size of each batch
        feature_dim: Dimension of feature vectors
        num_classes: Number of possible classes
        seed: Random seed for reproducibility

    Returns:
        A generator that yields (features, labels, weights) tuples
    """
    # Initialize the RNG key
    key = fastmath.random.get_prng(seed)

    while True:
        # Split the key for this iteration to get two independent random keys
        key, subkey1, subkey2 = fastmath.random.split(key, 3)

        # Generate features, labels and weights
        features = fastmath.random.randint(subkey1, (batch_size, feature_dim), minval=0, maxval=10_000)
        labels = fastmath.random.randint(subkey2, (batch_size,), minval=0, maxval=num_classes)
        weights = jnp.ones((batch_size,))

        # Yield the batch
        yield (features, labels, weights)


mode = "train"
model = tl.Serial(
    tl.Embedding(vocab_size=10_000, d_feature=1),
    Transpose(),
    tl.Dropout(rate=0.1, mode=mode),
    tl.LeakyRelu(a=0.1),
    tl.Dense(20, use_bias=False),
)

# CrossEntropyLossWithLogSoftmax() make overhead ner 6 second in comparison to pure execution sequence of ore.LogSoftmax(), _CrossEntropy(), _WeightedMean(),
# When we use gpu the result is near 40-50 second the cpu near 60-65 difference 10-15 second per operation
# Accelerated version improve computation to cuda:0 time: 11.7531 seconds cpu - TFRT_CPU_0 time: 64.3705 seconds
model_with_loss = tl.Serial(model, CrossEntropyLossWithLogSoftmax())

model_with_loss_accelerated = tl.Accelerate(model_with_loss)

batch_generator = create_batch_generator(batch_size=32)
example_batch = next(batch_generator)

# Initialize model
print("Initializing model...")
init_start = time.time()
_, _ = model_with_loss_accelerated.init(shapes.signature(example_batch))
init_time = time.time() - init_start
print(f"Model initialization time: {init_time:.4f} seconds")

device = fastmath.jax.jax.devices("cpu")[0]
with fastmath.jax.jax.default_device(device):
    start_time = time.time()
    for _ in range(5_000):
        example_batch = next(batch_generator)
        y = model_with_loss_accelerated(example_batch)
    cpu_time = time.time() - start_time
    print(f"{device} time: {cpu_time:.4f} seconds")
    print(y)


x = np.array([[[1, 2, 3]]])
transpose_layer = Transpose()
result = transpose_layer(x)
transpose_layer_accelerated = tl.Accelerate(transpose_layer)
result = transpose_layer_accelerated(x)

# Define a sample computation
def compute():
    # Get a random key
    key = fastmath.random.get_prng(0)

    result = jnp.zeros((4_000, 20))

    for _ in range(100):
        key, subkey1, subkey2 = fastmath.random.split(key, 3)

        x = fastmath.random.normal(subkey1, (4_000, 10_000))
        y = fastmath.random.normal(subkey2, (10_000, 20))

        result = jnp.dot(x, y)

    return result


# Run on CPU
cpu_device = fastmath.jax.jax.devices("cpu")[0]
with fastmath.jax.jax.default_device(cpu_device):
    start_time = time.time()
    compute().block_until_ready()
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")

# Run on GPU
gpu_device = fastmath.jax.jax.devices("gpu")[0]
with fastmath.jax.jax.default_device(gpu_device):
    start_time = time.time()
    compute().block_until_ready()
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f} seconds")

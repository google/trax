"""Machine learning utilities for dataset handling, model training and evaluation."""

import time

from enum import Enum
from typing import Callable, Dict, Generator, Optional, Tuple, Union

import datasets
import numpy as np

from absl import logging
from sklearn.datasets import load_digits, load_iris

import trax.fastmath as fastmath

from trax.fastmath import numpy as jnp
from trax.utils import shapes

# Set global logging verbosity
logging.set_verbosity(logging.INFO)

# Constants
DEFAULT_BATCH_SIZE = 32
LOG_INTERVAL = 10


class DeviceType(Enum):
    """Supported device types for computation."""
    CPU = "cpu"
    GPU = "gpu"


class Dataset(Enum):
    """Supported datasets."""
    IRIS = "iris"
    DIGITS = "digits"
    MNIST = "mnist"


class Splits(Enum):
    """Supported datasets."""
    TRAIN = "train"
    TEST = "test"


def load_mnist(split: str = Splits.TRAIN.value) -> Tuple[np.ndarray, np.ndarray]:
    # Load the MNIST dataset using Hugging Face Datasets
    # Use 'mnist' for the standard MNIST dataset
    dataset = datasets.load_dataset("mnist", split=split)

    # Pre-allocate arrays with the correct shape
    num_examples = len(dataset)
    X = np.zeros((num_examples, 784), dtype=np.float32)
    y = np.zeros(num_examples, dtype=np.int64)

    # Process each example in the dataset
    i = 0
    for image, label in zip(dataset['image'], dataset['label']):
        # Flatten image from (28, 28) to (784,) and normalize
        X[i] = np.array(image).reshape(-1).astype(np.float32) / 255.0
        y[i] = label
        i += 1

    return X, y


def load_dataset(
    dataset_name: str = Dataset.IRIS.value,
    split: str = Splits.TRAIN.value,
) -> Union[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a dataset by name and split.

    Args:
        dataset_name: Name of the dataset to load.
        split: Which split to load ('train', 'test', or 'validation')

    Returns:
        For sklearn datasets: Tuple of (data, labels) arrays.
        For TensorFlow datasets: A TensorFlow dataset object.
    """
    if dataset_name == Dataset.IRIS.value:
        dataset = load_iris()
        data, labels = dataset.data, dataset.target
        # For sklearn datasets, we'll simulate train/test split
        if split == 'test':
            # Use last 20% as test
            test_size = len(data) // 5
            return data[-test_size:], labels[-test_size:]
        else:
            # Use first 80% as train
            train_size = len(data) - (len(data) // 5)
            return data[:train_size], labels[:train_size]

    elif dataset_name == Dataset.DIGITS.value:
        dataset = load_digits()
        data, labels = dataset.data, dataset.target
        # For sklearn datasets, we'll simulate train/test split
        if split == 'test':
            # Use last 20% as test
            test_size = len(data) // 5
            return data[-test_size:], labels[-test_size:]
        else:
            # Use first 80% as train
            train_size = len(data) - (len(data) // 5)
            return data[:train_size], labels[:train_size]

    elif dataset_name == Dataset.MNIST.value:
        x, y = load_mnist(split=split)
        return  x, y
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")



def create_batch_generator(
    data: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: Optional[int] = None
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Create an infinite generator that produces shuffled batches.

    Args:
        data: data array, shape [n_examples, n_data].
        labels: Labels array, shape [n_examples].
        weights: Optional sample weights array, shape [n_examples]. If None, uses all ones.
        batch_size: Number of samples per batch.
        seed: Random seed for reproducibility.

    Yields:
        A tuple (data_batch, labels_batch, weights_batch).
    """
    n_samples = data.shape[0]

    # Convert inputs to arrays and prepare weights if needed
    data = np.asarray(data)
    labels = np.asarray(labels)
    weights = np.ones_like(labels) if weights is None else np.asarray(weights)

    # Initialize random number generator and shuffle indices
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    batch_index = 0
    while True:
        # Get batch indices with wraparound handling
        end_index = batch_index + batch_size
        if end_index <= n_samples:
            batch_indices = indices[batch_index:end_index]
        else:
            overflow = end_index - n_samples
            batch_indices = np.concatenate([indices[batch_index:], indices[:overflow]], axis=0)
            rng.shuffle(indices)

        # Yield batch data converted to jax arrays
        yield (
            np.array(data[batch_indices]),
            np.array(labels[batch_indices]),
            np.array(weights[batch_indices]),
        )

        # Update index for next batch
        batch_index = (batch_index + batch_size) % n_samples
        if batch_index == 0:
            rng.shuffle(indices)


def initialize_model(model_with_loss, example_batch) -> Tuple[float, float]:
    """
    Initialize and compile a model using an example batch.

    Args:
        model_with_loss: Model with loss function to initialize.
        example_batch: Example batch for initialization.

    Returns:
        Tuple of (initialization_time, compilation_time) in seconds.
    """
    logging.info("Initializing model...")
    init_start = time.time()
    _, _ = model_with_loss.init(shapes.signature(example_batch))
    init_time = time.time() - init_start
    logging.info(f"Model initialization time: {init_time:.4f} seconds")

    logging.info("Compiling model with first batch...")
    compile_start = time.time()
    _ = model_with_loss(example_batch)
    compile_time = time.time() - compile_start
    logging.info(f"Compilation time: {compile_time:.4f} seconds")

    return init_time, compile_time


def _get_target_device(device_type: str):
    """Helper function to get the target device."""
    if device_type == DeviceType.CPU.value:
        return fastmath.devices(DeviceType.CPU.value)[0]
    elif device_type == DeviceType.GPU.value:
        return fastmath.devices(DeviceType.GPU.value)[0]
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


def train_model(
    trainer,
    batch_generator: Callable,
    num_steps: int,
    base_rng,
    device_type: str = DeviceType.CPU.value
) -> list:
    """
    Train a model for a specified number of steps.

    Args:
        trainer: The model trainer.
        batch_generator: Generator that produces training batches.
        num_steps: Number of training steps.
        base_rng: Base random number generator.
        device_type: Type of device to use for training ("cpu" or "gpu").

    Returns:
        List of loss values for each training step.
    """
    logging.info(f"\n\n{'='*20} RUNNING ON {device_type.upper()} {'='*20}")
    logging.info(
        f"Backend: {fastmath.backend_name()}, Global devices: {fastmath.global_device_count()}"
    )

    losses = []
    training_start = time.time()

    # Set target device via context
    target_device = _get_target_device(device_type)

    with fastmath.jax.jax.default_device(target_device):
        for step in range(num_steps):
            step_start = time.time()
            step_rng, base_rng = fastmath.random.split(base_rng)
            batch = next(batch_generator)
            loss = trainer.one_step(batch, step_rng, step=step)
            step_time = time.time() - step_start
            losses.append(loss)

            # Log progress at regular intervals
            if step % LOG_INTERVAL == 0 or step == num_steps - 1:
                logging.info(f"Step {step}, Loss: {loss:.4f}, Step time: {step_time:.4f} sec")

    # Print training summary
    training_time = time.time() - training_start
    avg_step_time = training_time / num_steps
    logging.info(f"Total training time: {training_time:.4f} sec, Average step: {avg_step_time:.4f} sec")

    return losses


def compute_accuracy(predicted: jnp.ndarray, true_labels: jnp.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predicted: 1D array of integer class predictions, shape [N].
        true_labels: 1D array of integer ground-truth labels, shape [N].

    Returns:
        Accuracy as a float between 0 and 1.
    """
    return jnp.mean(predicted == true_labels)


def evaluate_model(
    trainer,
    test_data,
    test_labels,
    device_type: str = DeviceType.CPU.value,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    num_batches: int = 100
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        trainer: The trained model trainer.
        test_data: Test data array.
        test_labels: Test labels array.
        device_type: Type of device to use for evaluation.
        batch_size: Batch size for evaluation.
        seed: Random seed for batch generation.
        num_batches: Number of batches to evaluate.

    Returns:
        Dictionary with evaluation metrics including accuracy and mean loss.
    """
    logging.info(f"\n\n{'='*20} EVALUATING MODEL {'='*20}")

    # Create batch generator for test data
    test_batch_gen = create_batch_generator(
        test_data, test_labels, None, batch_size, seed
    )

    # Set up evaluation environment
    target_device = _get_target_device(device_type)
    dummy_rng = fastmath.random.get_prng(10)

    # Initialize evaluation metrics
    total_loss = 0.0
    total_accuracy = 0.0

    # Evaluate model on test set
    with fastmath.jax.jax.default_device(target_device):
        for i in range(num_batches):
            batch = next(test_batch_gen)

            # Get model predictions
            predictions = trainer.model_with_loss.sublayers[0](
                batch,
                weights=trainer.model_with_loss.sublayers[0].weights,
                state=trainer.model_with_loss.sublayers[0].state,
                rng=dummy_rng,
            )

            # Calculate accuracy
            predicted = jnp.argmax(predictions[0], axis=1)
            labels = predictions[1]
            batch_accuracy = compute_accuracy(predicted, labels)
            total_accuracy += batch_accuracy

            # Calculate loss
            batch_loss = trainer.model_with_loss(batch, rng=dummy_rng)
            total_loss += batch_loss

            # Log progress
            if i % LOG_INTERVAL == 0 or i == num_batches - 1:
                logging.info(f"Test batch {i}, Accuracy: {batch_accuracy:.4f}, Loss: {batch_loss:.4f}")

    # Calculate final metrics
    mean_accuracy = total_accuracy / num_batches
    mean_loss = total_loss / num_batches

    # Log summary
    logging.info("\nTest results:")
    logging.info(f"  Mean accuracy: {mean_accuracy:.4f}")
    logging.info(f"  Mean loss: {mean_loss:.4f}")

    return {
        "accuracy": float(mean_accuracy),
        "loss": float(mean_loss)
    }

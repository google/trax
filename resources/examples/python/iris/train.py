import trax.fastmath as fastmath

from resources.examples.python.base import (
    Dataset,
    DeviceType,
    compute_accuracy,
    create_batch_generator,
    initialize_model,
    load_dataset,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.fastmath import numpy as jnp
from trax.trainers import jax as trainers


def build_model():
    # Build your model with loss function
    model = tl.Serial(
        tl.Dense(16, use_bias=True),
        tl.Relu(),
        tl.Dense(3, use_bias=False)
    )
    model_with_loss = tl.Serial(model, tl.CrossEntropyLossWithLogSoftmax())
    return model_with_loss


def main():
    # Load data
    X, y = load_dataset(Dataset.IRIS.value)
    batch_gen = create_batch_generator(X, y, batch_size=50, seed=42)
    example_batch = next(batch_gen)

    # Build and initialize model
    model_with_loss = build_model()
    initialize_model(model_with_loss, example_batch)

    # Setup optimizer and trainers
    optimizer = optimizers.SGD(0.1)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)
    num_steps = 5000
    # Run training on CPU and/or GPU
    losses = train_model(trainer, batch_gen, num_steps, base_rng, device_type=DeviceType.GPU.value)

    with fastmath.jax.jax.default_device(DeviceType.CPU.value):
        dummy_rng = fastmath.random.get_prng(10)

        predictions = trainer.model_with_loss.sublayers[0](
            example_batch,
            weights=trainer.model_with_loss.sublayers[0].weights,
            state=trainer.model_with_loss.sublayers[0].state,
            rng=dummy_rng,
        )

        predicted = jnp.argmax(predictions[0], axis=1)
        labels = predictions[1]
        print(f"Accuracy: {compute_accuracy(predicted, labels)}")

        mean_loss = jnp.mean(jnp.array(losses))
        print(f"Mean loss: {mean_loss}")


if __name__ == "__main__":
    main()

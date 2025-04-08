import trax.fastmath as fastmath

from resources.examples.python.base import (
    Dataset,
    DeviceType,
    Splits,
    create_batch_generator,
    evaluate_model,
    initialize_model,
    load_dataset,
    train_model,
)
from trax import layers as tl
from trax import optimizers
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
    # Default setup
    DEFAULT_BATCH_SIZE = 8
    STEPS_NUMBER = 20_000

    # Load data
    X, y = load_dataset(Dataset.IRIS.value)
    batch_generator = create_batch_generator(X, y, batch_size=DEFAULT_BATCH_SIZE, seed=42)
    example_batch = next(batch_generator)

    # Build and initialize model
    model_with_loss = build_model()
    initialize_model(model_with_loss, example_batch)

    # Setup optimizer and trainers
    optimizer = optimizers.SGD(0.1)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)

    # Run training on CPU and/or GPU
    train_model(
        trainer, batch_generator, STEPS_NUMBER, base_rng, device_type=DeviceType.GPU.value
    )

    # Load test data
    test_data, test_labels = load_dataset(
        dataset_name=Dataset.IRIS.value, split=Splits.TEST.value
    )

    # Evaluate model on test set
    test_results = evaluate_model(
        trainer=trainer,
        test_data=test_data,
        test_labels=test_labels,
        device_type=DeviceType.CPU.value,
        batch_size=DEFAULT_BATCH_SIZE,
        num_batches=100,
    )

    print(f"Final test accuracy: {test_results['accuracy']:.4f}")




if __name__ == "__main__":
    main()

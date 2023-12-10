from typing import Tuple, Dict, List
import mlflow
import mlflow.tensorflow
from datetime import datetime
from tensorflow import keras
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)
from keras.models import Sequential
from keras.utils import plot_model
import tensorflow as tf
from load_fer2013 import load_fer2013, preprocess


def setup_mlflow() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    experiment_name = "Baseline"
    experiment_description = (
        "This is a neural network for classifiying human emotions based on facial expressions."
        "This experiment will create a baseline neural network for further experiments."
    )
    experiment_tags = {
        "project_name": "facial-emotion-recognition",
        "experiment_name": experiment_name,
        "dataset": "fer2013",
        "mlflow.note.content": experiment_description,
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(experiment_tags)
    mlflow.tensorflow.autolog()


def create_model(
    input_shape: Tuple[int, int, int], num_classes: int, params
) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(),
            BatchNormalization(),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", keras.metrics.CategoricalAccuracy()],
    )
    return model


import tensorflow as tf
from typing import Tuple

def load_and_preprocess_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    data = load_fer2013()
    num_classes = 7

    # Define splits for train, validation, and test sets
    split_train = int(len(data) * 0.7)
    split_test = int(len(data) * 0.1)
    split_val = len(data) - split_train - split_test

    # Create a TensorFlow dataset from the data
    dataset = tf.data.Dataset.from_tensor_slices(dict(data))
    dataset = dataset.map(
        lambda row: preprocess(row, num_classes), num_parallel_calls=tf.data.AUTOTUNE
    )

    # Partition the data into train, validation, and test sets
    train_dataset = (
        dataset.take(split_train).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        dataset.skip(split_train).take(split_val).batch(32).prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        dataset.skip(split_train + split_val).batch(32).prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset


def train_and_log_model(
    model: Sequential,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    params: Dict[str, str | int | List[str]],
) -> None:
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params["epochs"],  # type: ignore
        batch_size=params["batch_size"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=params["early_stopping_patience"],  # type: ignore
                # restore_best_weights=True, Removing this stops the model from being far worse at the last step... Dont know why
            ),
            keras.callbacks.ModelCheckpoint("./output/best_model", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(
                factor=params["lr_reduction_factor"], patience=params["lr_patience"]  # type: ignore
            ),
        ],
    )
    model.save("./output/emotion.h5")
    mlflow.log_params(params)
    plot_model(model, to_file="./output/model.png", show_shapes=True)
    mlflow.log_artifact("./output/model.png")
    model.save_weights("./output/model_weights/model_weights")
    mlflow.log_artifact("./output/model_weights")


if __name__ == "__main__":
    setup_mlflow()
    with mlflow.start_run() as run:
        input_shape = (48, 48, 1)
        num_classes = 7
        params = {
            "batch_size": 128,
            "epochs": 50,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "early_stopping_patience": 5,
            "learning_rate": 0.0001,
            "lr_reduction_factor": 0.1,
            "lr_patience": 3,
        }

        train_dataset, val_dataset, test_dataset = load_and_preprocess_data()
        model = create_model(input_shape, num_classes, params)
        run = train_and_log_model(model, train_dataset, val_dataset, params)

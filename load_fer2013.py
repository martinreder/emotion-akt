import os
import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple


def load_fer2013() -> pd.DataFrame:
    """Load the emotion dataset as a tf.data.Dataset."""
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output(
            "curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz",
            shell=True,
        )
    print("Loading dataset...")
    data = pd.read_csv("fer2013/fer2013.csv")
    return data


def preprocess(row, num_classes):
    # Convert the 'pixels' tensor to string and split
    pixel_string = row["pixels"]
    pixel_values = tf.strings.split([pixel_string], sep=" ")
    pixel_values = tf.strings.to_number(pixel_values, out_type=tf.int32)

    # Convert the RaggedTensor to a regular tensor
    pixel_values = tf.RaggedTensor.to_tensor(pixel_values, default_value=0)

    # Reshape and normalize the pixel values
    pixels = tf.reshape(pixel_values, (48, 48, 1))
    pixels = tf.cast(pixels, tf.float32) / 255.0

    # Prepare the label
    emotion = tf.one_hot(row["emotion"], depth=num_classes)

    return pixels, emotion

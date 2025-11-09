
import tensorflow as tf
from tensorflow import keras
def load_data():
    train_data =  tf.keras.utils.image_dataset_from_directory(
    '../dataset/train',  # relative path from your script
    image_size=(48, 48),
    batch_size=32,  #take a batch of size in data contains 32 img
    label_mode="categorical",
    shuffle=True,
    color_mode="grayscale"
    )
    test_data= tf.keras.utils.image_dataset_from_directory(
    '../dataset/test',
    image_size=(48,48),
    label_mode="categorical",
    color_mode="grayscale",
    )
    return train_data, test_data


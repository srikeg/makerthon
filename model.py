import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from torchvision import datasets, models
import tensorflow as tf
import keras
from tensorflow.keras import layers

pretrained_model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=4,
    classifier_activation="softmax",
    name="resnet50",
)

pretrained_model.trainable = True

inputs = keras.Input(shape=(224, 224, 3))
x = pretrained_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(4, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


IMAGES_DIR = "/home/steffi/Downloads/Beispiel_Source_Code/speckle/images"

ds = tf.keras.utils.image_dataset_from_directory(
    IMAGES_DIR,
    image_size=(224,244),
    batch_size=32,
    shuffle=True
)

embed()

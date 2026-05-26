import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


IMAGES_DIR = "/home/steffi/Downloads/Beispiel_Source_Code/speckle/images"

ds = tf.keras.utils.image_dataset_from_directory(
    IMAGES_DIR,
    image_size=(224,224),
    batch_size=32,
    shuffle=True,
    seed=42
)

total_batches = len(ds)
train_size = int(0.7 * total_batches)
val_size = int(0.15 * total_batches)

# Use .take() and .skip() to partition the dataset
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size).take(val_size)
test_ds = ds.skip(train_size + val_size)

# --- 4. PERFORMANCE OPTIMIZATION ---
# Cache and prefetch data to prevent I/O bottlenecks during training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. TRAINING ---
EPOCHS = 10  # Adjust as needed

print("\nStarting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 6. EVALUATION ---
print("\nEvaluating on Test Dataset:")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

embed()

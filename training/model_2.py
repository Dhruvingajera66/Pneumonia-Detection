import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
import collections
import os
from sklearn.utils import class_weight


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "../data/train", seed=123, shuffle=True, batch_size=32, image_size=(256, 256)
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "../data/test", seed=123, shuffle=True, batch_size=32, image_size=(256, 256)
)

train_batches = int(0.8 * len(dataset))
val_batches = len(dataset) - train_batches

train_dataset = dataset.take(train_batches)
val_dataset = dataset.skip(train_batches)

class_names = dataset.class_names
num_classes = len(class_names)


class_counts = collections.defaultdict(int)
for _, labels in dataset:
    for label in labels.numpy():
        class_counts[class_names[label]] += 1

labels = np.concatenate([labels.numpy() for _, labels in dataset])
class_weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(labels), y=labels
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ]
)


def preprocess(image, label):
    image = tf.image.resize(image, (256, 256)) / 255.0
    return image, label


train_ds = (
    train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_ds = val_dataset.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_dataset.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(256, 256, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    verbose=1,
    class_weight=class_weight_dict,
)

scores = model.evaluate(test_ds)
print(f"Test Accuracy: {scores[1] * 100:.2f}%")

model.save("../final_model/Version_2.h5")

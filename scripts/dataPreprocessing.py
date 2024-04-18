import os

import tensorflow as tf

base_dir = 'D:/ICBT/cloth2/data/images/'
train_dir = os.path.join(base_dir, 'train/', 'images')
val_dir = os.path.join(base_dir, 'val/', 'images')
test_dir = os.path.join(base_dir, 'test/', 'images')

img_height, img_width = 224, 224
batch_size = 32


def preprocess_image(image_path):
    # Read the image from file
    image = tf.io.read_file(image_path)
    # Decode the JPEG image to uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize the image to the desired size
    image = tf.image.resize(image, [img_height, img_width])
    # Scale the image to [0, 1]
    image = image / 255.0

    return image


def preprocess_labels(label_path):
    label = tf.strings.split(label_path, os.path.sep)[-1]
    return label


def load_and_preprocess_image(image_path):
    return preprocess_image(image_path), preprocess_labels(image_path)

train_dataset = tf.data.Dataset.list_files(os.path.join(train_dir, '*.jpg'))
val_dataset = tf.data.Dataset.list_files(os.path.join(val_dir, '*.jpg'))
test_dataset = tf.data.Dataset.list_files(os.path.join(test_dir, '*.jpg'))

train_dataset = train_dataset.map(load_and_preprocess_image)
val_dataset = val_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.map(load_and_preprocess_image)

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


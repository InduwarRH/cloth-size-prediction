import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Sequential

# Define paths
data_base_dir = 'D:/ICBT/cloth2/data/'
annotations_file_path = os.path.join(data_base_dir, 'annotation/list_landmarks.txt')
partitions_file_path = os.path.join(data_base_dir, 'annotation/list_eval_partition.txt')
model_save_dir = 'D:/ICBT/cloth2/model/'

# Read the annotations
annotations = pd.read_csv(annotations_file_path, sep='\\s+', skiprows=1,
                          names=['image_name', 'clothes_type', 'variation_type', 'landmark_visibility_1',
                                 'landmark_location_x_1', 'landmark_location_y_1', 'landmark_visibility_2',
                                 'landmark_location_x_2', 'landmark_location_y_2', 'landmark_visibility_3',
                                 'landmark_location_x_3', 'landmark_location_y_3', 'landmark_visibility_4',
                                 'landmark_location_x_4', 'landmark_location_y_4', 'landmark_visibility_5',
                                 'landmark_location_x_5', 'landmark_location_y_5', 'landmark_visibility_6',
                                 'landmark_location_x_6', 'landmark_location_y_6', 'landmark_visibility_7',
                                 'landmark_location_x_7', 'landmark_location_y_7', 'landmark_visibility_8',
                                 'landmark_location_x_8', 'landmark_location_y_8'])

# Read the partitions
partitions = pd.read_csv(partitions_file_path, sep='\\s+', header=None, names=['image_name', 'partition'])
print(partitions.head())
# Merge annotations with partitions
merged_data = annotations.merge(partitions, on='image_name')

img_height, img_width = 224, 224
num_landmarks = 8  # Update this if you have a different number of landmarks


# Function to preprocess images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0
    return image


# Function to load data and preprocess from the merged dataframe
def load_and_preprocess_from_dataframe(image_name, data):
    print( data['partition'])
    image_path = tf.strings.join([data_base_dir, 'images/', data['partition'], '/images/', image_name], separator='')
    image = preprocess_image(image_path)

    landmarks = [data[f'landmark_location_x_{i}'] for i in range(1, num_landmarks + 1)] + \
                [data[f'landmark_location_y_{i}'] for i in range(1, num_landmarks + 1)]
    landmarks = tf.convert_to_tensor(landmarks, dtype=tf.float32)

    return image, landmarks


# Model definition
model = Sequential([Input(shape=(img_height, img_width, 3)), Conv2D(16, (3, 3), activation='relu'), MaxPooling2D(2, 2),
                    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D(2, 2), Flatten(), Dense(128, activation='relu'),
                    Dense(num_landmarks * 2)  # x and y coordinates for each landmark
                    ])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Convert the merged DataFrame to a TensorFlow dataset
data_dict = {key: tf.constant(value) for key, value in merged_data.items()}
dataset = tf.data.Dataset.from_tensor_slices((data_dict['image_name'], data_dict))
dataset = dataset.map(load_and_preprocess_from_dataframe).batch(32).prefetch(tf.data.AUTOTUNE)

# Fit the model
model.fit(dataset, epochs=10)

# Save the model
model.save(model_save_dir)
print("Model trained and saved at " + model_save_dir)

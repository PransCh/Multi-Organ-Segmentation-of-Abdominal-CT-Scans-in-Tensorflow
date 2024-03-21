import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
  images = sorted(glob(os.path.join(path, "3D-images", "*")))
  masks = sorted(glob(os.path.join(path, "masks", "*")))
    
  split_size = int(split * len(images))

  train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
  train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

  train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
  train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

  return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def get_colormap(path):
#     mat_path = os.path.join(path, "/kaggle/input/finaldataset/ultra-pro-max-3D/colors.mat")
    colormap = scipy.io.loadmat("/kaggle/input/finaldataset/ultra-pro-max-3D/colors.mat")["colors"]
    # Give your respective .mat file path

    classes = [
        "Background",
        "Aorta",
        "Left kidney",
        "Right kidney",
        "Gallbladder",
        "Liver",
        "Pancreas",
        "Spleen",
        "Stomach"
    ]
    return classes, colormap

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = x.astype(np.float32)
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))

    #Masl processing
    output=[]
    for color in COLORMAP:
      cmap = np.all(np.equal(x, color), axis=-1)
      # cv2.imwrite(f"cmap_{i}.png", cmap * 255)
      output.append(cmap)

    output = np.stack(output, axis =-1)
    output = output.astype(np.uint8)

    return output

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x =read_image(x)
        y=read_mask(y)

        return x,y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
    image.set_shape([IMG_H, IMG_W, 3])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])

    return image, mask

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    IMG_H = 256
    IMG_W = 256
    NUM_CLASSES = 9
    input_shape = (IMG_H, IMG_W, 3)

    batch_size = 8
    lr = 1e-4
    num_epochs = 50
    dataset_path = "/kaggle/input/finaldataset/ultra-pro-max-3D/Prepared-data"
    # Give your respective dataset path

    model_path = os.path.join("files", "Unet_model.keras")
    csv_path = os.path.join("files", "Unet_data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")

    """ Process the colormap """
    CLASSES, COLORMAP = get_colormap(dataset_path)

    """ Dataset Pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

#     """ TPU Configuration """
#     resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(resolver)
#     tf.tpu.experimental.initialize_tpu_system(resolver)

#     strategy = tf.distribute.experimental.TPUStrategy(resolver)

#     with strategy.scope():
    """ Model """
    model = build_unet(input_shape, NUM_CLASSES)
    # model.load_weights(model_path)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )
    # model.summary()

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=num_epochs,
        callbacks=callbacks
    )
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, multilabel_confusion_matrix
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.losses import SparseCategoricalCrossentropy

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

""" Load and split the dataset """
def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "Training", "img_folder", "*")))[:5000]
    masks = sorted(glob(os.path.join(path, "Training", "mask_folder", "*")))[:5000]

    split_size = int(split * len(images))

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def get_colormap(path):
    mat_path = os.path.join(path, "ultimate.mat")
    colormap = scipy.io.loadmat(mat_path)["color"]
    # Give the appropriate .mat file

    classes = [
        "Background",
        "Spleen",
        "Right kidney",
        "Left kidney",
        "Liver",
        "Gallbladder",
        "Stomach",
        "Aorta",
        "Inferior vena cava",
        "Portal vein",
        "Pancreas"
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

    output=[]
    for color in COLORMAP:
        cmap = np.all(np.equal(x, color), axis=-1)
        output.append(cmap)

    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)

    return output

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y

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

def grayscale_to_rgb(mask, classes, colormap):
    h, w, _ = mask.shape
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image, mask, pred, save_image_path):
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    assert image.shape == mask.shape
    assert image.shape == pred.shape

    alpha = 0.5
    blended_image1 = alpha * image + (1 - alpha) * mask
    blended_image2 = alpha * image + (1 - alpha) * pred

    cat_images = np.concatenate([image, line, mask, line, pred, line, blended_image1, blended_image2], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")
    create_dir("results2/predictions")

    IMG_H = 320
    IMG_W = 416
    NUM_CLASSES = 11
    dataset_path = "/kaggle/input/utdata/Ultimate_dataset-20231206T103555Z-001/Ultimate_dataset" 
    # Give your respective dataset path
    model_path = "/kaggle/input/model-h5/Unet_100_model.h5"
    # Give your respective model name

    CLASSES, COLORMAP = get_colormap(dataset_path)

    model = tf.keras.models.load_model(model_path)

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}")
    print("")

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = x.split("/")[-1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMG_W, IMG_H))
        image_x = image
        image = np.expand_dims(image, axis=0)

        mask = cv2.imread(y, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (IMG_W, IMG_H))
        mask_x = mask
        onehot_mask = []
        for color in COLORMAP:
            cmap = np.all(np.equal(mask, color), axis=-1)
            onehot_mask.append(cmap)
        onehot_mask = np.stack(onehot_mask, axis=-1)
        onehot_mask = np.argmax(onehot_mask, axis=-1)
        onehot_mask = onehot_mask.astype(np.int32)

        pred = model.predict(image, verbose=0)[0]
        pred = np.argmax(pred, axis=-1)
        pred = pred.astype(np.float32)

        save_image_path = f"results2/predictions/{name}.png"
        save_results(image_x, mask_x, pred, save_image_path)

        onehot_mask = onehot_mask.flatten()
        pred = pred.flatten()

        labels = [i for i in range(NUM_CLASSES)]

        f1_value = f1_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
        jac_value = jaccard_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
        

        SCORE.append([f1_value, jac_value])

    score = np.array(SCORE)
    score = np.mean(score, axis=0)

    accuracy = accuracy_score(onehot_mask, pred.flatten())

    f = open("files/score.csv", "w")
    f.write("Class,F1,Jaccard,Accuracy\n")

    l = ["Class", "F1", "Jaccard", "Dice", "Accuracy"]
    print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s} {l[3]:10s} {l[4]:10s}")
    print("-" * 60)

    for i in range(score.shape[1]):
        class_name = CLASSES[i]
        f1 = score[0, i]
        jac = score[1, i]
        dice = score[2, i]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f} - {dice:1.5f} - {accuracy:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f},{dice:1.5f},{accuracy:1.5f}\n")

    print("-" * 60)
    class_mean = np.mean(score, axis=-1)
    class_name = "Mean"
    f1 = class_mean[0]
    jac = class_mean[1]
    dice = class_mean[2]
    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f} - {dice:1.5f} - {accuracy:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f},{dice:1.5f},{accuracy:1.5f}\n")

    f.close()
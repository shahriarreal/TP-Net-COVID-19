import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os, sys
from keras.models import load_model, Model
import csv
from keras.utils import to_categorical


def load_image(root_path, height, width, classes):
    label_map = {'BacterialPneumonia': 3,
                 'COVID-19': 0,
                 'Normal': 2,
                 'ViralPneumonia': 1}
    data, labels = [], []
    dirs = os.listdir(root_path)

    for d in dirs:
        if d not in label_map:
            continue
        print(d)
        path = os.path.join(root_path, d)
        img_count = os.listdir(path)
        for img in img_count:
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((width, height))
                data.append(np.array(size_image))
                labels.append(label_map[d])
            except AttributeError:
                print(" ")

    Cells = np.array(data)
    labels = np.array(labels)

    s = np.arange(Cells.shape[0])
    np.random.seed(classes)
    np.random.shuffle(s)
    Cells = Cells[s]
    labels = labels[s]

    X = Cells.astype('float32') / 255
    y = to_categorical(labels, classes)
    return X, y


def data_processing(data_path, height, width, classes):
    #train_path = os.path.join(data_path, 'NonAugmentedTrain')
    val_path = os.path.join(data_path, 'ValData')
    # X_train, y_train = load_image(train_path, height, width, classes)
    X_train, y_train = '', ''
    X_val, y_val = load_image(val_path, height, width, classes)
    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    height = 400
    width = 300
    channels = 3
    classes = 4
    save_path = './training_info.xlsx'
    pre_trained_model_path = ''
    data_path = './covid19-detection-xray-dataset'

    model_path = './test_model.h5'  # abs path to .h5 file
    model = load_model(model_path, compile=False)
    eval_thresh = 0.5

    _, _, X_val, y_val = data_processing(data_path, height, width, classes)

    tp_count = 0
    eval_result = model.predict(X_val)
    total_count = len(eval_result)
    for i, img in enumerate(eval_result):
        if max(img) < eval_thresh:
            result = classes
        else:
            result = np.argmax(img)
        if result == np.argmax(y_val[i]):
            tp_count += 1
    print('Accuracy:', str(tp_count / total_count))
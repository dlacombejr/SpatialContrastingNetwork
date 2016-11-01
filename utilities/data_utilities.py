import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10


def load_data(n_train=4000):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_valid, y_valid) = (None, None)
    if n_train < X_train.shape[0]:
        (X_train, y_train), (X_valid, y_valid) = train_valid_split(X_train, y_train, n_train=n_train)
        y_train = np_utils.to_categorical(np.asarray(y_train, dtype='int16'))
        y_valid = np_utils.to_categorical(np.asarray(y_valid, dtype='int16'))
        y_test = np_utils.to_categorical(np.asarray(y_test, dtype='int16'))
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def train_valid_split(X_train, y_train, n_train=4000):

    classes = list(np.unique(y_train))
    n_classes = len(classes)
    examples, c, h, w = X_train.shape
    n_valid = examples - n_train
    train_samples_per_class = n_train / n_classes
    valid_samples_per_class = n_valid / n_classes
    X_train_ = np.zeros((n_train, c, h, w))
    X_valid_ = np.empty((n_valid, c, h, w))
    y_train_ = np.zeros((n_train, 1))
    y_valid_ = np.zeros((n_valid, 1))

    for class_ in sorted(classes):

        class_index = np.where(y_train == class_)[0]
        X_class = X_train[class_index]
        y_class = y_train[class_index]

        train_start = train_samples_per_class * class_
        train_end = train_start + train_samples_per_class
        valid_start = valid_samples_per_class * class_
        valid_end = valid_start + valid_samples_per_class

        X_train_[train_start:train_end] = X_class[:train_samples_per_class]
        y_train_[train_start:train_end] = y_class[:train_samples_per_class]
        X_valid_[valid_start:valid_end] = X_class[:valid_samples_per_class]
        y_valid_[valid_start:valid_end] = y_class[:valid_samples_per_class]

    return (X_train_, y_train_), (X_valid_, y_valid_)

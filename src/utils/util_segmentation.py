import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from keras.metrics import Precision
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.metrics import Recall
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model



def _get_mask_slices_indexes(z, n_max=1):
    count = 0
    z[len(z) - 1] = 0 if z[len(z) - 1] == 1 else z[len(z) - 1]
    areas = np.zeros(len(z))
    indexes = np.zeros((len(z), 2))

    val_i_1 = z[0]
    for i, val_i in enumerate(z):
        change01 = val_i_1 == 0 and val_i == 1
        change10 = val_i_1 == 1 and val_i == 0

        indexes[count, 0] = indexes[count, 0] * (not change01) + i * change01
        indexes[count, 1] = indexes[count, 1] * (not change10) + i * change10
        areas[count] = indexes[count, 1] - indexes[count, 0]

        val_i_1 = val_i

        count += change10

    areas = areas[areas != 0]
    np.bincount(areas.astype(int))
    n_max = min(n_max, len(areas) - 1)
    max_idxs = np.argpartition(areas, -n_max)[-n_max:]

    return indexes[max_idxs]


def get_max_box(x_min, x_max, y_min, y_max):
    if x_min.__len__() == 0:
        print("No bounding box found")
    return min(x_min), max(x_max), min(y_min), max(y_max)



def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)

    y_true = tf.cast(y_true, tf.float32)  ##

    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def load_saved_model(path):
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(path)

    return model

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape, nFilter):
    inputs = Input(input_shape)

    # nFilter = 16 #64
    s1, p1 = encoder_block(inputs, nFilter)
    s2, p2 = encoder_block(p1, nFilter * 2)
    s3, p3 = encoder_block(p2, nFilter * 4)
    s4, p4 = encoder_block(p3, nFilter * 8)

    b1 = conv_block(p4, nFilter * 16)

    d1 = decoder_block(b1, s4, nFilter * 8)
    d2 = decoder_block(d1, s3, nFilter * 4)
    d3 = decoder_block(d2, s2, nFilter * 2)
    d4 = decoder_block(d3, s1, nFilter)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    pass

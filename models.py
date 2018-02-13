#!/usr/bin/env python3
"""Implement several encoder classes"""

from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from keras.preprocessing import image
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Concatenate
import numpy as np


class ResNet:
    __name__ = "ResNet"

    def __init__(self, batch_size=32):
        self.model = ResNet50(include_top=False, weights='imagenet', pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_resnet(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


class Inception:
    __name__ = "Inception"

    def __init__(self, batch_size=32):
        self.model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_inception(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


class VGG:
    __name__ = "VGG"

    def __init__(self, batch_size=32):
        model = VGG16(include_top=False, weights="imagenet", pooling="avg")
        x2 = GlobalAveragePooling2D()(model.get_layer("block2_conv2").output)  # 128
        x3 = GlobalAveragePooling2D()(model.get_layer("block3_conv3").output)  # 256
        x4 = GlobalAveragePooling2D()(model.get_layer("block4_conv3").output)  # 512
        x5 = GlobalAveragePooling2D()(model.get_layer("block5_conv3").output)  # 512
        x = Concatenate()([x2, x3, x4, x5])
        self.model = Model(inputs=model.input, outputs=x)
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_vgg(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


if __name__ == "__main__":
    img_path = "data/elephant.jpg"
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_inception(x)

    model = InceptionV3()
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print("Predicted:", decode_predictions(preds, top=3)[0])
    # Predicted: [('n02504013', 'Indian_elephant', 0.78864819), ('n01871265', 'tusker', 0.029346621), ('n02504458', 'African_elephant', 0.01768155)]


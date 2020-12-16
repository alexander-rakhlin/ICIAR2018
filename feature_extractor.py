#!/usr/bin/env python3
"""Extract deep CNN features from a set of images and dump them as Numpy arrays image_file_name.npy"""

import argparse
import numpy as np
import cv2
from scipy import ndimage
from os.path import basename, join, exists
from os import makedirs
from threaded_generator import threaded_generator
from time import time
import sys
np.random.seed(13)

PATCH_SIZES = [400, 650]
SCALES = [0.5]

DEFAULT_INPUT_DIR = "data/train"
DEFAULT_PREPROCESSED_ROOT = "data/preprocessed/train"

PATCHES_PER_IMAGE = 20
AUGMENTATIONS_PER_IMAGE = 50
COLOR_LO = 0.7
COLOR_HI = 1.3
BATCH_SIZE = 16     # decrease if necessary

NUM_CACHED = 160


def recursive_glob(root_dir, file_template="*.tif"):
    """Traverse directory recursively. Starting with Python version 3.5, the glob module supports the "**" directive"""

    if sys.version_info[0] * 10 + sys.version_info[1] < 35:
        import fnmatch
        import os
        matches = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, file_template):
                matches.append(os.path.join(root, filename))
        return matches
    else:
        import glob
        return glob.glob(root_dir + "/**/" + file_template, recursive=True)


def normalize_staining(img):
    """
    Adopted from "Classification of breast cancer histology images using Convolutional Neural Networks",
    Teresa Araújo , Guilherme Aresta, Eduardo Castro, José Rouco, Paulo Aguiar, Catarina Eloy, António Polónia,
    Aurélio Campilho. https://doi.org/10.1371/journal.pone.0177544

    Performs staining normalization.

    # Arguments
        img: Numpy image array.
    # Returns
        Normalized Numpy image array.
    """
    Io = 240
    beta = 0.15
    alpha = 1
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape(h * w, c)
    OD = -np.log((img.astype("uint16") + 1) / Io)
    ODhat = OD[(OD >= beta).all(axis=1)]
    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

    Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = HE.T
    Y = OD.reshape(h * w, c).T

    C = np.linalg.lstsq(HE, Y)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

    return Inorm


def hematoxylin_eosin_aug(img, low=0.7, high=1.3, seed=None):
    """
    "Quantification of histochemical staining by color deconvolution"
    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf

    Performs random hematoxylin-eosin augmentation

    # Arguments
        img: Numpy image array.
        low: Low boundary for augmentation multiplier
        high: High boundary for augmentation multiplier
    # Returns
        Augmented Numpy image array.
    """
    D = np.array([[1.88, -0.07, -0.60],
                  [-1.02, 1.13, -0.48],
                  [-0.55, -0.13, 1.57]])
    M = np.array([[0.65, 0.70, 0.29],
                  [0.07, 0.99, 0.11],
                  [0.27, 0.57, 0.78]])
    Io = 240

    h, w, c = img.shape
    OD = -np.log10((img.astype("uint16") + 1) / Io)
    C = np.dot(D, OD.reshape(h * w, c).T).T
    r = np.ones(3)
    r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
    img_aug = np.dot(C * r, M)

    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
    return img_aug


def zoom_aug(img, zoom_var, seed=None):
    """Performs a random spatial zoom of a Numpy image array.

    # Arguments
        img: Numpy image array.
        zoom_var: zoom range multiplier for width and height.
        seed: Random seed.
    # Returns
        Zoomed Numpy image array.
    """
    scale = np.random.RandomState(seed).uniform(low=1 / zoom_var, high=zoom_var)
    resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized_img


def get_crops(img, size, n, seed=None):
    """Creates random square crops of given size from a Numpy image array. No rotation added

    # Arguments
        img: Numpy image array.
        size: size of crops.
        n: number of crops
        seed: Random seed.
    # Returns
        Numpy array of crops, shape (n, size, size, c).
    """
    h, w, c = img.shape
    assert all([size < h, size < w])
    crops = []
    for _ in range(n):
        top = np.random.randint(low=0, high=h - size + 1)
        left = np.random.randint(low=0, high=w - size + 1)
        crop = img[top: top + size, left: left + size].copy()
        crop = np.rot90(crop, np.random.randint(low=0, high=4))
        if np.random.random() > 0.5:
            crop = np.flipud(crop)
        if np.random.random() > 0.5:
            crop = np.fliplr(crop)
        crops.append(crop)

    crops = np.stack(crops)
    assert crops.shape == (n, size, size, c)
    return crops


def get_crops_free(img, size, n, seed=None):
    """Creates random square crops of given size from a Numpy image array. With rotation

    # Arguments
        img: Numpy image array.
        size: size of crops.
        n: number of crops
        seed: Random seed.
    # Returns
        Numpy array of crops, shape (n, size, size, c).
    """
    h, w, c = img.shape
    assert all([size < h, size < w])
    d = int(np.ceil(size / np.sqrt(2)))
    crops = []
    for _ in range(n):
        center_y = np.random.randint(low=0, high=h - size + 1) + size // 2
        center_x = np.random.randint(low=0, high=w - size + 1) + size // 2
        m = min(center_y, center_x, h - center_y, w - center_x)
        if m < d:
            max_angle = np.pi / 4 - np.arccos(m / d)
            top = center_y - m
            left = center_x - m
            precrop = img[top: top + 2 * m, left: left + 2 * m]
        else:
            max_angle = np.pi / 4
            top = center_y - d
            left = center_x - d
            precrop = img[top: top + 2 * d, left: left + 2 * d]

        precrop = np.rot90(precrop, np.random.randint(low=0, high=4))
        angle = np.random.uniform(low=-max_angle, high=max_angle)
        precrop = ndimage.rotate(precrop, angle * 180 / np.pi, reshape=False)

        precrop_h, precrop_w, _ = precrop.shape
        top = (precrop_h - size) // 2
        left = (precrop_w - size) // 2
        crop = precrop[top: top + size, left: left + size]

        if np.random.random() > 0.5:
            crop = np.flipud(crop)
        if np.random.random() > 0.5:
            crop = np.fliplr(crop)
        crops.append(crop)

    crops = np.stack(crops)
    assert crops.shape == (n, size, size, c)
    return crops


def norm_pool(features, p=3):
    """Performs descriptor pooling

    # Arguments
        features: Numpy array of descriptors.
        p: degree of pooling.
    # Returns
        Numpy array of pooled descriptor.
    """
    return np.power(np.power(features, p).mean(axis=0), 1/p)


def encode(crops, model):
    """Encodes crops

    # Arguments
        crops: Numpy array of crops.
        model: Keras encoder.
    # Returns
        Numpy array of pooled descriptor.
    """
    features = model.predict(crops)
    pooled_features = norm_pool(features)
    return pooled_features


def process_image(image_file):
    """Extract multiple crops from a single image

    # Arguments
        image_file: Path to image.
    # Yields
        Numpy array of image crops.
    """
    img = cv2.imread(image_file)
    if SCALE != 1:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = normalize_staining(img)

    for _ in range(AUGMENTATIONS_PER_IMAGE):
        img_aug = hematoxylin_eosin_aug(img_norm, low=COLOR_LO, high=COLOR_HI)
        # img_aug = zoom_aug(img_aug, ZOOM_VAR)

        # single_image_crops = get_crops_free(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)
        single_image_crops = get_crops(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)
        yield single_image_crops


def crops_gen(file_list):
    """Generates batches of crops from image list, one augmentation a time

    # Arguments
        file_list: List of image files.
    # Yields
        Tuple of Numpy array of image crops and name of the file.
    """
    for i, (image_file, output_file) in enumerate(file_list):
        print("Crops generator:", i + 1)
        for crops in process_image(image_file):
            yield crops, output_file


def features_gen(crops_and_output_file, model):
    """Processes crop generator, encodes them and dumps pooled descriptors

    # Arguments
        crops_and_output_file: generator of crops and file names.
        model: Keras encoder.
    # Returns: None
    """
    ts = time()
    current_file = None
    pooled_features = []
    i = 0
    for j, (crops, output_file) in enumerate(crops_and_output_file):
        if current_file is None:
            current_file = output_file
        features = encode(crops, model)
        if output_file == current_file:
            pooled_features.append(features)
        else:
            np.save(current_file, np.stack(pooled_features))
            pooled_features = [features]
            current_file = output_file
            average_time = int((time() - ts) / (i + 1))
            print("Feature generator: {}, {} sec/image.".format(i + 1, average_time))
            i += 1
    if len(pooled_features) > 0:
        np.save(current_file, np.stack(pooled_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--images",
        required=False,
        default=DEFAULT_INPUT_DIR,
        metavar="img_dir",
        help="Input image directory. Default: data/train")
    arg("--features",
        required=False,
        default=DEFAULT_PREPROCESSED_ROOT,
        metavar="feat_dir",
        help="Feature root dir. Default: data/preprocessed/train")
    args = parser.parse_args()
    INPUT_DIR = args.images
    PREPROCESSED_ROOT = args.features

    from models import ResNet, Inception, VGG
    NN_MODELS = [ResNet, Inception, VGG]

    input_files = recursive_glob(INPUT_DIR)

    for SCALE in SCALES:
        print("SCALE:", SCALE)
        for NN_MODEL in NN_MODELS:
            print("NN_MODEL:", NN_MODEL.__name__)
            for PATCH_SZ in PATCH_SIZES:
                print("PATCH_SZ:", PATCH_SZ)
                PREPROCESSED_PATH = join(PREPROCESSED_ROOT, "{}-{}-{}".format(NN_MODEL.__name__, SCALE, PATCH_SZ))
                if not exists(PREPROCESSED_PATH):
                    makedirs(PREPROCESSED_PATH)

                model = NN_MODEL(batch_size=BATCH_SIZE)

                output_files = [join(PREPROCESSED_PATH, basename(f).replace("tif", "npy")) for f in input_files]
                file_list = zip(input_files, output_files)

                crops_and_output_file = crops_gen(file_list)
                crops_and_output_file_ = threaded_generator(crops_and_output_file, num_cached=NUM_CACHED)
                features_gen(crops_and_output_file_, model)

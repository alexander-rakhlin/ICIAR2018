#!/usr/bin/env python3
"""Generates submission"""
import pickle
from os.path import join
import numpy as np
from utils import load_data
import pandas as pd
import argparse


MODELS_DIR = "models/LGBMs"
DEFAULT_PREPROCESSED_ROOT = "data/preprocessed/test/"
DEFAULT_SUBMISSION_FILE = "submission/submission.csv"

N_FOLDS = 10
N_SEEDS = 5
N_CLASSES = 4
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
AUGMENTATIONS_PER_IMAGE = 50

MODELS = [
    "ResNet-0.5-400",
    "ResNet-0.5-650",
    "VGG-0.5-400",
    "VGG-0.5-650",
    "Inception-0.5-400",
    "Inception-0.5-650",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--features",
        required=False,
        default=DEFAULT_PREPROCESSED_ROOT,
        metavar="feat_dir",
        help="Feature root dir. Default: data/preprocessed/test")
    arg("--submission",
        required=False,
        default=DEFAULT_SUBMISSION_FILE,
        metavar="submission",
        help="Submission file. Default: submission/submission.csv")
    args = parser.parse_args()
    PREPROCESSED_ROOT = args.features
    SUBMISSION_FILE = args.submission

    scores = []
    files = None
    len_x = None
    for fold in range(N_FOLDS):
        for model_name in MODELS:
            name, scale, crop = model_name.split("-")
            for seed in range(N_SEEDS):
                x, fl = load_data(join(PREPROCESSED_ROOT, "{}-{}-{}".format(name, scale, crop)))
                if files is None:
                    files = fl
                    len_x = len(x)
                else:
                    np.testing.assert_array_equal(fl, files)
                    assert len(x) == len_x
                model_file = "lgbm-{}-{}-{}-f{}-s{}.pkl".format(name, scale, crop, fold, seed)
                with open(join(MODELS_DIR, name, model_file), "rb") as f:
                    model = pickle.load(f)

                sc = model.predict(x)
                sc = sc.reshape(-1, AUGMENTATIONS_PER_IMAGE, N_CLASSES)
                scores.append(sc)

    scores = np.stack(scores)   # N_FOLDS*N_MODELS*N_SEEDS x N x AUGMENTATIONS_PER_IMAGE x N_CLASSES
    scores = scores.mean(axis=(0, 2))
    y_pred = np.argmax(scores, axis=1)
    labels = [CLASSES[i] for i in y_pred]

    df = pd.DataFrame(list(zip(map(lambda s: s.replace(".npy", ".tif"), files), labels)), columns=["image", "label"])
    df = df.sort_values("image")
    df.to_csv(SUBMISSION_FILE, header=False, index=False)

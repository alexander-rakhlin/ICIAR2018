#!/usr/bin/env python3
"""Utilities for cross-validation.
Notice data/folds-10.pkl we use in 10-fold cross-val. Keep it to replicate our results"""

import numpy as np
import glob
from os.path import basename, join
from sklearn.model_selection import StratifiedKFold
import pickle


def load_data(in_dir, folds=None, split=None):
    """Builds train/test data from preprocessed features for a given split

    # Arguments
        in_dir: Input directory containing *.npy CNN feature files.
        folds: None or list of splits dict{
                                             "train": {
                                                         "x": train files list,
                                                         "y": train labels},
                                             "test": {
                                                         "x": test files list,
                                                         "y": test labels}}
                                        }
        split: None or split number.
    # Returns
        Tran/test data (features and labels) for a given split, if `folds` is not None
        Test data (only features) and file names, if `folds` is None
    """
    if folds:
        y_train = []
        x_train = []
        for f, l in zip(folds[split]["train"]["x"], folds[split]["train"]["y"]):
            x = np.load(join(in_dir, f))
            x_train.append(x)
            y_train.append([l] * len(x))
        x_train = np.vstack(x_train)
        y_train = np.concatenate(y_train)

        y_test = []
        x_test = []
        for f, l in zip(folds[split]["test"]["x"], folds[split]["test"]["y"]):
            x = np.load(join(in_dir, f))
            x_test.append(x)
            y_test.append([l] * len(x))
        x_test = np.vstack(x_test)
        y_test = np.concatenate(y_test)

        return x_train, y_train, x_test, y_test
    else:
        files = glob.glob(in_dir + "/*.npy")
        x = []
        for f in files:
            x.append(np.load(f))
        return np.vstack(x), np.array([basename(f) for f in files])


def make_folds():
    """Creates stratified splits based on train directory listing

    # Dumps
        folds: list of splits dict{
                                   "train": {
                                               "x": train files list,
                                               "y": train labels},
                                   "test": {
                                               "x": test files list,
                                               "y": test labels}}
                                }
    """
    files = np.array([basename(f) for f in glob.glob("data/preprocessed/train/ResNet-0.5-400/*.npy")])
    labels = []
    classes = np.array([0, 1, 2, 3])
    for f in files:
        lb = np.array([f.startswith("n"),
                       f.startswith("b"),
                       f.startswith("is"),
                       f.startswith("iv")])
        labels.append(classes[np.argmax(lb)])
    labels = np.array(labels)

    folds = []
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(files, labels):
        f_train, f_test = files[train_index], files[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        folds.append({"train": {"x": f_train, "y": y_train}, "test": {"x": f_test, "y": y_test}})

    with open("data/folds-10.pkl", "wb") as f:
        pickle.dump(folds, f)

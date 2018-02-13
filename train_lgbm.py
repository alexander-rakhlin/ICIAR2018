#!/usr/bin/env python3
"""Trains LightGBM models on various features, data splits. Dumps models and predictions"""

import pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from utils import load_data
from os.path import join, exists
from os import makedirs
import argparse


CROP_SIZES = [400, 650]
SCALES = [0.5]
NN_MODELS = ["ResNet", "Inception", "VGG"]

AUGMENTATIONS_PER_IMAGE = 50
NUM_CLASSES = 4
RANDOM_STATE = 1
N_SEEDS = 5
VERBOSE_EVAL = False
with open("data/folds-10.pkl", "rb") as f:
    FOLDS = pickle.load(f)

LGBM_MODELS_ROOT = "models/LGBMs"
CROSSVAL_PREDICTIONS_ROOT = "predictions"
DEFAULT_PREPROCESSED_ROOT = "data/preprocessed/train"


def _mean(x, mode="arithmetic"):
    """
    Calculates mean probabilities across augmented data

    # Arguments
        x: Numpy 3D array of probability scores, (N, AUGMENTATIONS_PER_IMAGE, NUM_CLASSES)
        mode: type of averaging, can be "arithmetic" or "geometric"
    # Returns
        Mean probabilities 2D array (N, NUM_CLASSES)
    """
    assert mode in ["arithmetic", "geometric"]
    if mode == "arithmetic":
        x_mean = x.mean(axis=1)
    else:
        x_mean = np.exp(np.log(x + 1e-7).mean(axis=1))
        x_mean = x_mean / x_mean.sum(axis=1, keepdims=True)
    return x_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--features",
        required=False,
        default=DEFAULT_PREPROCESSED_ROOT,
        metavar="feat_dir",
        help="Feature root dir. Default: data/preprocessed/train")
    args = parser.parse_args()
    PREPROCESSED_ROOT = args.features

    learning_rate = 0.1
    num_round = 70
    param = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": ["multi_logloss", "multi_error"],
        "verbose": 0,
        "learning_rate": learning_rate,
        "num_leaves": 191,
        "feature_fraction": 0.46,
        "bagging_fraction": 0.69,
        "bagging_freq": 0,
        "max_depth": 7,
    }

    for SCALE in SCALES:
        print("SCALE:", SCALE)
        for NN_MODEL in NN_MODELS:
            print("NN_MODEL:", NN_MODEL)
            for CROP_SZ in CROP_SIZES:
                print("PATCH_SZ:", CROP_SZ)
                INPUT_DIR = join(PREPROCESSED_ROOT, "{}-{}-{}".format(NN_MODEL, SCALE, CROP_SZ))
                acc_all_seeds = []
                for seed in range(N_SEEDS):
                    accuracies = []
                    for fold in range(len(FOLDS)):
                        feature_fraction_seed = RANDOM_STATE + seed * 10 + fold
                        bagging_seed = feature_fraction_seed + 1
                        param.update({"feature_fraction_seed": feature_fraction_seed, "bagging_seed": bagging_seed})

                        print("Fold {}/{}, seed {}".format(fold + 1, len(FOLDS), seed))
                        x_train, y_train, x_test, y_test = load_data(INPUT_DIR, FOLDS, fold)
                        train_data = lgb.Dataset(x_train, label=y_train)
                        test_data = lgb.Dataset(x_test, label=y_test)
                        gbm = lgb.train(param, train_data, num_round, valid_sets=[test_data], verbose_eval=VERBOSE_EVAL)

                        # pickle model
                        model_file = "lgbm-{}-{}-{}-f{}-s{}.pkl".format(NN_MODEL, SCALE, CROP_SZ, fold, seed)
                        model_root = join(LGBM_MODELS_ROOT, NN_MODEL)
                        if not exists(model_root):
                            makedirs(model_root)
                        with open(join(model_root, model_file), "wb") as f:
                            pickle.dump(gbm, f)

                        scores = gbm.predict(x_test)
                        scores = scores.reshape(-1, AUGMENTATIONS_PER_IMAGE, NUM_CLASSES)
                        preds = {
                            "files": FOLDS[fold]["test"]["x"],
                            "y_true": y_test,
                            "scores": scores,
                        }
                        preds_file = "lgbm_preds-{}-{}-{}-f{}-s{}.pkl".format(NN_MODEL, SCALE, CROP_SZ,
                                                                              fold, seed)
                        preds_root = join(CROSSVAL_PREDICTIONS_ROOT, NN_MODEL)
                        if not exists(preds_root):
                            makedirs(preds_root)
                        with open(join(preds_root, preds_file), "wb") as f:
                            pickle.dump(preds, f)

                        mean_scores = _mean(scores, mode="arithmetic")
                        y_pred = np.argmax(mean_scores, axis=1)
                        y_true = y_test[::AUGMENTATIONS_PER_IMAGE]
                        acc = accuracy_score(y_true, y_pred)
                        print("Accuracy:", acc)
                        accuracies.append(acc)

                    acc_seed = np.array(accuracies).mean()  # acc of a seed
                    acc_all_seeds.append(acc_seed)
                    print("{}-{}-{} Accuracies: [{}], mean {:5.3}".format(NN_MODEL, SCALE, CROP_SZ,
                                                                          ", ".join(map(lambda s: "{:5.3}".format(s), accuracies)),
                                                                          acc_seed))
                print("Accuracy of all seeds {:5.3}".format(np.array(acc_all_seeds).mean()))


"""
ResNet-1.0-800
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[0.875, 0.725, 0.875, 0.875,   0.8,  0.75,   0.8,  0.85,  0.85,  0.85], mean 0.825
[0.875, 0.775, 0.825, 0.875, 0.775,  0.75,   0.8,  0.85,  0.85, 0.825], mean  0.82
[0.875,  0.75,  0.85,  0.85,   0.8,  0.75, 0.725, 0.875, 0.875, 0.875], mean 0.823
[0.875,  0.75,  0.85,  0.85,   0.8, 0.775, 0.775, 0.825,  0.85, 0.825], mean 0.817
[0.875,  0.75, 0.875, 0.875, 0.775, 0.775, 0.825,  0.85,   0.9,  0.85], mean 0.835
Accuracy of all seeds 0.824

ResNet-1.0-1300
learning_rate = 0.1, 60 steps [0.9, 0.775, 0.825, 0.875, 0.8, 0.775, 0.8, 0.875, 0.85, 0.85]
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[0.925,  0.75,  0.85,  0.85,   0.8, 0.725,   0.8, 0.825,   0.9,  0.85], mean 0.828
[0.875,  0.75,   0.8, 0.875,  0.75, 0.725,   0.8, 0.825, 0.875,  0.85], mean 0.812
[0.875,  0.75,   0.8, 0.825, 0.775, 0.725, 0.825,  0.85,  0.95, 0.875], mean 0.825
[  0.9, 0.775,   0.8,  0.85, 0.725, 0.725,   0.8,   0.8, 0.875, 0.825], mean 0.807
[ 0.85, 0.725,   0.8, 0.825, 0.725,  0.75,   0.8,   0.8, 0.875,  0.85], mean   0.8
Accuracy of all seeds 0.815

VGG-1.0-800
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[ 0.85, 0.775, 0.825, 0.875,  0.85, 0.775,   0.8,   0.8,  0.85,   0.8], mean  0.82
[ 0.85,  0.75,  0.85, 0.875, 0.825, 0.775, 0.775, 0.825, 0.875, 0.775], mean 0.818
[ 0.85,  0.75, 0.825, 0.875,  0.85,  0.75, 0.825, 0.825, 0.875,  0.75], mean 0.818
[ 0.85,   0.8, 0.825, 0.875, 0.825,   0.8,   0.8,   0.8, 0.875, 0.775], mean 0.823
[0.825, 0.775, 0.775, 0.875,  0.85, 0.775,   0.8,   0.8,  0.85, 0.725], mean 0.805
Accuracy of all seeds 0.816

VGG-1.0-1300
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[0.825,  0.75,   0.8, 0.875, 0.825,   0.8,   0.8,  0.75,  0.85,   0.8], mean 0.807
[0.825,  0.75, 0.825,   0.9,   0.8, 0.825, 0.725, 0.825, 0.875, 0.775], mean 0.812
[ 0.85,   0.8,   0.8, 0.875, 0.775, 0.775, 0.775, 0.775, 0.875, 0.775], mean 0.808
[  0.8, 0.775, 0.775,   0.9,   0.8,   0.8,   0.8, 0.825, 0.875, 0.775], mean 0.812
[0.825, 0.825,   0.8,   0.9,   0.8,   0.8, 0.725,   0.8,  0.85,   0.8], mean 0.812
Accuracy of all seeds  0.81


ResNet-0.5-650
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[ 0.95, 0.775, 0.875,   0.9, 0.825, 0.775,  0.85,  0.85,  0.85, 0.825], mean 0.847
[  0.9, 0.775, 0.875,   0.9, 0.775,   0.7, 0.875,   0.8,  0.85, 0.825], mean 0.828
[  0.9, 0.775,  0.85,   0.9, 0.825,  0.75, 0.875, 0.825, 0.825, 0.825], mean 0.835
[0.875, 0.775,  0.85, 0.875,   0.8, 0.725,  0.85, 0.825,  0.85, 0.825], mean 0.825
[0.925, 0.775,  0.85,   0.9, 0.825,  0.75, 0.825,  0.85,  0.85, 0.825], mean 0.838
Accuracy of all seeds 0.834


ResNet-0.5-400
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[0.925, 0.825, 0.875, 0.875, 0.775, 0.825, 0.825, 0.825, 0.825, 0.825], mean 0.84
[0.925, 0.775, 0.875, 0.875,   0.8,  0.85,  0.85,   0.8,  0.85, 0.825], mean 0.842
[  0.9, 0.725, 0.875, 0.875, 0.825,  0.85, 0.875,  0.85,  0.85, 0.825], mean 0.845
[0.925, 0.775, 0.875, 0.875,   0.8,  0.85,  0.85,  0.85,  0.85, 0.825], mean 0.847
[0.925, 0.775, 0.825, 0.875, 0.775, 0.825,  0.85, 0.825, 0.825, 0.825], mean 0.833
acc_all_seeds 0.841


VGG-0.5-400
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7  
[ 0.85,  0.85,   0.8, 0.825, 0.825,   0.8, 0.825,  0.85, 0.875, 0.825], mean 0.832
[0.875, 0.825,  0.85,  0.85, 0.875,  0.85,   0.8, 0.825, 0.875, 0.825], mean 0.845
[  0.9, 0.825, 0.825,  0.85,  0.85,   0.8,   0.8,  0.85, 0.875,  0.85], mean 0.842
[0.875, 0.825,   0.8,  0.85, 0.825, 0.875,   0.8, 0.775, 0.875, 0.825], mean 0.832
[0.875, 0.825,   0.8, 0.825, 0.825,   0.8,   0.8,   0.8, 0.875, 0.825], mean 0.825
acc_all_seeds 0.835

VGG-0.5-650
learning_rate = 0.1, 70 steps
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[  0.9, 0.825, 0.775,  0.85, 0.825,   0.8, 0.825, 0.875,   0.9,   0.8], mean 0.838
[  0.9, 0.825, 0.775,  0.85,   0.8,   0.8,   0.8, 0.875,   0.9, 0.775], mean  0.83
[  0.9, 0.875, 0.825,  0.85,   0.8,  0.75,   0.8, 0.875,   0.9, 0.825], mean  0.84
[0.875,  0.85,  0.75,  0.85,   0.8,  0.75, 0.825,   0.8, 0.875,   0.8], mean 0.818
[  0.9,   0.9,   0.8,  0.85, 0.825,   0.8, 0.825,  0.85, 0.875, 0.825], mean 0.845
acc_all_seeds 0.8342


Inception_adv-1.0-1300
best acc reached so far 0.67, acc 0.7875, knobs num_leaves 121, feature_fraction 0.34, bagging_fraction 0.69, max_depth 38
learning_rate = 0.1, 60 steps [0.8, 0.85, 0.7375, 0.7875, 0.7375]
learning_rate = 0.1, 60 steps [0.8, 0.85, 0.8, 0.875, 0.8, 0.75, 0.75, 0.8, 0.8, 0.65]

Inception-0.5-650
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
[  0.9, 0.825,  0.75,   0.9,  0.85,   0.8,   0.8,  0.85, 0.775, 0.775], mean 0.823
[0.925,   0.8, 0.725,   0.9,  0.85,   0.8, 0.825, 0.825,   0.8,  0.75], mean  0.82
[  0.9, 0.825,  0.75,   0.9, 0.825, 0.825,  0.85,  0.85, 0.775, 0.775], mean 0.828
[0.925,   0.9, 0.725,   0.9, 0.825, 0.825, 0.825,  0.85,  0.75,  0.75], mean 0.828
[  0.9, 0.875, 0.725,   0.9,  0.85,   0.8,   0.8,  0.85,   0.8,   0.8], mean  0.83
Accuracy of all seeds 0.826

Inception-0.5-400
* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7
(no data)
"""

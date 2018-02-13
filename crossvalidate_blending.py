#!/usr/bin/env python3
"""Do cross-validation, output and dump statistics in data/roc_scores.pkl,
data/conf_mx.pkl, submission/crossvalidation.csv"""

import pickle
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from utils import load_data
import pandas as pd
import argparse


PREDS_DIR = "predictions/"
N_FOLDS = 10
N_SEEDS = 5
DEFAULT_N_CLASSES = 4

USE_PREDICTIONS = True
LGBM_MODELS_ROOT = "models/LGBMs"
DEFAULT_PREPROCESSED_ROOT = "data/preprocessed/train/"
with open("data/folds-10.pkl", "rb") as f:
    folds = pickle.load(f)
AUGMENTATIONS_PER_IMAGE = 50

models = [
    "ResNet-0.5-400",
    "ResNet-0.5-650",
    "VGG-0.5-400",
    "VGG-0.5-650",
    "Inception-0.5-400",
    "Inception-0.5-650",
]


def combine_model_scores(scores, y=None, cross_val=True):
    """
    Combine predictions across multiple models and augmentations and return labels.
    By default we use simple average without optimization on predicted labels (y=None)

    Arguments
        scores: Numpy array of probabilities, (n_models x n_samples x augmentations x n_classes)
        y: true labels, (n_samples,). Simple average if None. Logistic regression if given.
        cross_val: when y is given, blend via one-out cross-validation if True. Blend in-sample if False.
    Returns
        y_pred: labels, (n_samples,)
    """
    if y is None:
        return np.argmax(scores.mean(axis=(0, 2)), axis=1), scores.mean(axis=(0, 2))
    else:
        n_models, n_samples, n_aug, n_classes = scores.shape
        y = np.repeat(y[:, None], n_aug, axis=1)
        lr = LogisticRegression()
        if cross_val:
            pred = np.zeros((len(y), n_classes))
            for i in range(n_samples):
                idx = [k for k in range(len(y)) if k != i]
                x_train = scores[:, idx, ...].transpose(1, 2, 3, 0).reshape((n_samples - 1) * n_aug,
                                                                            n_models * n_classes)
                y_train = y[idx].flatten()
                x_test = scores[:, i, ...].transpose(1, 2, 0).reshape(n_aug, n_models * n_classes)
                lr.fit(x_train, y_train)
                pred[i] = lr.predict_proba(x_test).mean(axis=0)
        else:
            x = scores.transpose(1, 2, 3, 0).reshape(n_samples * n_aug, n_models * n_classes)
            lr.fit(x, y.flatten())
            pred = lr.predict_proba(x)
            pred = pred.reshape(n_samples, n_aug, n_classes).mean(axis=1)
        return np.argmax(pred, axis=1), pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--features",
        required=False,
        default=DEFAULT_PREPROCESSED_ROOT,
        metavar="",
        help="Feature root dir. Default: data/preprocessed/train")
    arg("--n_classes",
        required=False,
        default=DEFAULT_N_CLASSES,
        type=int, choices=[2, 4],
        metavar="",
        help="Number of classes. Can be 2 or 4. Default: 4")
    arg("-predict",
        action="store_true",
        default=False,
        help="Predict folds during blending, do not use pre-saved CV predictions")
    args = parser.parse_args()
    PREPROCESSED_ROOT = args.features
    N_CLASSES = args.n_classes
    USE_PREDICTIONS = not args.predict

    blended_scores = []
    y_true = []
    y_pred = []
    scores = []
    files = []
    verification = defaultdict(dict)
    for fold in range(N_FOLDS):
        y_true_f = []
        scores_f = []
        files_f = []
        for model_name in models:
            name, scale, crop = model_name.split("-")
            scores_seeded = []
            for seed in range(N_SEEDS):
                if USE_PREDICTIONS:
                    # use pre-saved predictions
                    preds_file = "lgbm_preds-{}-{}-{}-f{}-s{}.pkl".format(name, scale, crop, fold, seed)
                    with open(join(PREDS_DIR, name, preds_file), "rb") as f:
                        preds = pickle.load(f)
                else:
                    # predict during CV
                    model_file = "lgbm-{}-{}-{}-f{}-s{}.pkl".format(name, scale, crop, fold, seed)
                    with open(join(LGBM_MODELS_ROOT, name, model_file), "rb") as f:
                        model = pickle.load(f)
                    _, _, x_test, y_test = load_data(join(PREPROCESSED_ROOT, "{}-{}-{}".format(name, scale, crop)),
                                                     folds, fold)
                    sc = model.predict(x_test)
                    sc = sc.reshape(-1, AUGMENTATIONS_PER_IMAGE, DEFAULT_N_CLASSES)
                    preds = {
                        "files": folds[fold]["test"]["x"],
                        "y_true": y_test,
                        "scores": sc,
                    }
                n_samples, apm, _ = preds["scores"].shape  # apm ~ AUGMENTATIONS_PER_IMAGE
                scores_seeded.append(preds["scores"])  # N_SEEDS * (N x AUGMENTATIONS_PER_IMAGE x NUM_CLASSES))
                y_true_f.append(preds["y_true"][::apm])
                files_f.append(preds["files"])
            scores_seeded = np.stack(scores_seeded)  # (N_SEEDS x N x AUGMENTATIONS_PER_IMAGE x NUM_CLASSES)

            # used to compare with seeds x folds stats for a model obtained during training
            verification[model_name][fold] = scores_seeded  # (N_SEEDS x N x AUGMENTATIONS_PER_IMAGE x NUM_CLASSES)

            # average across seeds, we do not need them anymore
            scores_f.append(scores_seeded.mean(axis=0))  # (N x AUGMENTATIONS_PER_IMAGE x NUM_CLASSES)

        y_true_f = np.stack(y_true_f)
        assert np.all(y_true_f == y_true_f[0])
        files_f = np.stack(files_f)
        assert np.all(files_f == files_f[0])
        y_true.append(y_true_f[0])
        files.append(files_f[0])

        scores_f = np.stack(scores_f)
        y_pred_f, scores_f = combine_model_scores(scores_f, y=None, cross_val=True)
        y_pred.append(y_pred_f)
        scores.append(scores_f)

    if N_CLASSES == 2:
        for fold in range(N_FOLDS):
            # scores[i] = scores[i].reshape(-1, 2, 2).sum(axis=-1)
            scores[fold] = scores[fold].reshape(-1, 2, 2).max(axis=-1)
            scores[fold] = scores[fold] / scores[fold].sum(axis=-1, keepdims=True)
            y_pred[fold][y_pred[fold] == 1] = 0
            y_pred[fold][y_pred[fold] > 1] = 1
            y_true[fold][y_true[fold] == 1] = 0
            y_true[fold][y_true[fold] > 1] = 1

            for model_name in models:
                v = verification[model_name][fold]
                N = v.shape[1]
                v = v.reshape(N_SEEDS, N, AUGMENTATIONS_PER_IMAGE, 2, 2).max(axis=-1)
                verification[model_name][fold] = v / v.sum(axis=-1, keepdims=True)

    # verify predictions
    all_models_folds = np.zeros((len(models), N_FOLDS))
    for m_i, model_name in enumerate(models):
        model_preds = []
        for fold in range(N_FOLDS):
            # (N_SEEDS x N x AUGMENTATIONS_PER_IMAGE x NUM_CLASSES)
            fold_preds = verification[model_name][fold].mean((0, 2)).argmax(-1)
            model_preds.append(fold_preds)
            all_models_folds[m_i, fold] = accuracy_score(y_true[fold], fold_preds)
        model_acc = accuracy_score(np.concatenate(y_true), np.concatenate(model_preds))
        print("{}: average across seeds: [{}]. All folds {}({:0.2})".
              format(model_name, ", ".join(map(lambda s: "{:5.3}".format(s), all_models_folds[m_i])),
                     model_acc, all_models_folds[m_i].std()))
    print()
    print("Std across models: [{}]. All folds {:5.3}".
          format(", ".join(map(lambda s: "{:5.3}".format(s), all_models_folds.std(0))), all_models_folds.std(0).mean()))
    acc_folded = np.array([accuracy_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])  # y_true, y_pred contain folds
    acc_blend = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    print("Blended model: [{}], mean {:5.3}, std {:5.3}".format(", ".join(map(lambda s: "{:5.3}".format(s), acc_folded)),
                                                                acc_blend, acc_folded.std()))

    # Dump crossvalidation stats
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    scores = np.concatenate(scores)
    files = np.concatenate(files)

    if N_CLASSES == 2:
        with open("data/roc_scores.pkl", "wb") as f:
            pickle.dump((scores, y_true), f)
    elif N_CLASSES == 4:
        with open("data/conf_mx.pkl", "wb") as f:
            pickle.dump((y_true, y_pred), f)
        CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
        labels = [CLASSES[i] for i in y_pred]
        df = pd.DataFrame(list(zip(map(lambda s: s.replace(".npy", ".tif"), files), labels)), columns=["image", "label"])
        df = df.sort_values("image")
        df.to_csv("submission/crossvalidation.csv", header=False, index=False)

"""
2-class Blended model: [ 0.95, 0.925,   0.9, 0.975, 0.925, 0.975, 0.925, 0.925,  0.95, 0.925], mean 0.938, std 0.023
4-class Blended model: [0.925, 0.825, 0.875, 0.875, 0.875,   0.9,  0.85, 0.875, 0.875,  0.85], mean 0.873, std 0.0261
"""

import numpy as np
import cv2
from scipy import ndimage

PATCHES_PER_IMAGE = 100
PATCH_SZ = 256
AUGMENTATIONS_PER_IMAGE = 100


def normalizeStaining(img):
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
    Quantification of histochemical staining by color deconvolution
    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf
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
    img_aug = np.dot(C, M) * r

    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
    return img_aug


def get_crops(img, size, n, seed=None):
    h, w, c = img.shape
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
    h, w, c = img.shape
    d = int(np.ceil(size / np.sqrt(2)))
    crops = []
    for _ in range(n):
        top = np.random.randint(low=0, high=h - size + 1) + size // 2
        left = np.random.randint(low=0, high=w - size + 1) + size // 2
        m = min(top, left, h - top, w - left)
        r = m / d
        if m < d:
            max_angle = np.pi / 4 - np.arccos(r)
            top -= m
            left -= m
            precrop = img[top: top + 2 * m, left: left + 2 * m].copy()
        else:
            max_angle = np.pi / 4
            top -= d
            left -= d
            precrop = img[top: top + 2 * d, left: left + 2 * d].copy()

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


def three_norm_pool(features):
    return np.power(np.power(features, 3).mean(axis=0), 1/3)


def encode(crops, model, preprocessor):
    crops = preprocessor(crops)
    features = model.predict(crops)
    pooled_features = three_norm_pool(features)
    return pooled_features


if __name__ == '__main__':

    img = cv2.imread("data/1.6484.4.tif")
    # img = cv2.imread("data/ICIAR2018_BACH_Challenge/Photos/Invasive/iv073.tif")
    # img = cv2.imread("data/ICIAR2018_BACH_Challenge/Photos/Normal/n068.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_norm = normalizeStaining(img)
    cv2.imwrite("data/1.6484.4.norm.tif", cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR))

    for i in range(5):
        img_aug = hematoxylin_eosin_aug(img_norm)
        cv2.imwrite("data/1.6484.4.aug{}.tif".format(i), cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

    crops = get_crops_free(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)
    for i, crop in enumerate(crops):
        cv2.imwrite("data/tmp/crop{}.tif".format(i), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

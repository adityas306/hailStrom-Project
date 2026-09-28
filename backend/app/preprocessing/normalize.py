import numpy as np


def minmax_normalize(
    data,
    min_value,
    max_value
):

    data = np.asarray(
        data,
        dtype=np.float32
    )

    result = (
        data - min_value
    ) / (
        max_value - min_value
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


def normalize_reflectivity(data):

    # Example radar range
    return minmax_normalize(
        data,
        -10.0,
        70.0
    )


def normalize_velocity(data):

    return minmax_normalize(
        data,
        -50.0,
        50.0
    )


def normalize_satellite(data):

    return minmax_normalize(
        data,
        180.0,
        330.0
    )


def normalize_lightning(data):

    return minmax_normalize(
        data,
        0.0,
        20.0
    )
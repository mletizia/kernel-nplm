import numpy as np

def standardize_dataset(feature, mean_REF=None, std_REF=None):
    feature = np.asarray(feature, dtype=np.float32)

    if mean_REF is None:
        mean = np.mean(feature, axis=0)
    else:
        mean = np.asarray(mean_REF, dtype=np.float32)

    if std_REF is None:
        std = np.std(feature, axis=0)
    else:
        std = np.asarray(std_REF, dtype=np.float32)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    feature_std = (feature - mean) / std

    return feature_std, mean, std
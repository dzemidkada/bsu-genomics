import numpy as np


def target_samples_mask(data, sample, loci):
    masks = [
        np.array(
            data[f'{locus}'].values == sample[f'{locus}'].values
        ).reshape(-1, 1)
        for locus in loci
    ]
    return np.hstack(masks)


def mask_threshold(x, threshold, reverse=False):
    return x * np.logical_xor(reverse, (x >= threshold))

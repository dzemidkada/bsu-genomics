import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def D(locations, point):
    return np.linalg.norm(locations - point, axis=1)


def D_f(locations, point, **kwargs):
    base_distance = D(locations, point)
    return np.exp(- base_distance * kwargs.get('d_alpha', 1))


def G(genotypes, genotype):
    return np.mean(genotypes == genotype, axis=1)


def G_f(genotypes, genotype, **kwargs):
    base_similarity = G(genotypes, genotype)
    return sigmoid(
        (base_similarity - kwargs.get('g_mean', 0.5)) * kwargs.get('g_std', 5))

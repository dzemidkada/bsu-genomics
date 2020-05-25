from collections import defaultdict

import numpy as np
import pandas as pd

from geolocation.distance import D


def collect_q_metrics(d_t, k, q, candidates):
    c_distances = D(candidates, q[['lat', 'long']].values)
    ce_distances = D(candidates, candidates.mean(axis=0))
    return pd.DataFrame({
        f'hit@{k}_{d_t}': [np.any(c_distances < d_t)],
        f'hit_count@{k}_{d_t}': [np.sum(c_distances < d_t)],
        f'min_dist@{k}_{d_t}': [np.min(c_distances)],
        f'median_dist@{k}_{d_t}:': [np.median(c_distances)],
        f'compactness@{k}_{d_t}': [np.mean(ce_distances)]
    })


def candidates_freq(C):
    result = defaultdict(int)
    for c in C:
        for cc in c:
            result[tuple(cc)] += 1
    return result


def collect_test_metrics(d_t, k, Q, C):
    # Local query-related metrics
    results = pd.concat([
        collect_q_metrics(d_t, k, Q.iloc[i:i + 1], C[i])
        for i in range(len(C))
    ]).mean(axis=0)
    # Global
    c_freq = candidates_freq(C)

    results['coverage'] = len(c_freq)
    results['mean_c_repeats'] = np.mean(list(v_ for k_, v_ in c_freq.items()))

    return results

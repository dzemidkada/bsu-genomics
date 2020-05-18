import numpy as np

from geolocation.config import TARGET_AREA
from geolocation.utils import mask_threshold, target_samples_mask


class CandidatePointsGenerator:
    def __init__(self, train_data, loci, linkage_agg=None):
        self._train_data = train_data.copy()
        self._loci = list(loci)
        self._linkage_agg = linkage_agg or np.sum

    def __generate_grid(self, grid_size):
        return [
            (x, y)
            for x in np.linspace(*TARGET_AREA[0], grid_size)
            for y in np.linspace(*TARGET_AREA[1], grid_size)
        ]

    def get_ranked_candidates(self, sample, hyperparams, n):
        grid_size = int(hyperparams['grid_size'])
        candidates_grid = self.__generate_grid(grid_size)

        filtered_candidates = list()
        for lat, long in candidates_grid:
            # Euclid distance
            distance = np.sqrt(
                (self._train_data.lat - lat)**2 +
                (self._train_data.long - long)**2
            )
            distance = np.exp(- distance * hyperparams['distance_sensitivity'])
            # Genotype intersections
            genotype_similarity = np.mean(
                target_samples_mask(
                    self._train_data, sample, self._loci),
                axis=1
            )
            point_signal = self._linkage_agg(
                mask_threshold(
                    distance, hyperparams['distance_threshold'], reverse=True
                ) * mask_threshold(
                    genotype_similarity, hyperparams['similarity_threshold'])
            )
            if point_signal:
                filtered_candidates.append((lat, long, point_signal))

        return sorted(
            filtered_candidates, key=lambda x: x[2], reverse=True
        )[:n]

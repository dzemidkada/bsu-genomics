from collections import Counter

import numpy as np

from geolocation.config import TARGET_AREA


class CandidatesGenerator:
    def __init__(self):
        pass

    def generate_candidates(self):
        raise NotImplementedError()


class GridGenerator(CandidatesGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        self._grid_size = kwargs.get('grid_size', 20)

    def generate_candidates(self):
        return np.array([
            (x, y)
            for x in np.linspace(*TARGET_AREA[0], self._grid_size)
            for y in np.linspace(*TARGET_AREA[1], self._grid_size)
        ])


class FrequentLocationsGenerator(CandidatesGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        self._k = kwargs.get('k', 400)
        self._train_data = kwargs.get('train_data', None)
        self.__preprocess()

    def __preprocess(self):
        assert self._train_data is not None
        available_locations = Counter()
        available_locations.update([
            (np.round(row.lat, 3),
             np.round(row.long, 3))
            for _, row in self._train_data.iterrows()
        ])
        self._most_freq_locations = np.array([
            k for k, v in available_locations.most_common(self._k)
        ])

    def generate_candidates(self):
        return self._most_freq_locations

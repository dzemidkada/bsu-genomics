import numpy as np

from geolocation.config import AVAILABLE_LOCI
from geolocation.distance import D, D_f, G, G_f


class RankingGeneratorBase:
    def __init__(self, train_data, **params):
        self._train_data = train_data
        self._params = params
        self.__preprocess_data()

    def __preprocess_data(self):
        self._locations = np.round(self._train_data[['lat', 'long']].values, 3)

    def rank_candidates(self, q, candidates):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def get_loci(self, q):
        return list(set(q.columns).intersection(AVAILABLE_LOCI))


class RankingGeneratorRandom(RankingGeneratorBase):
    def __init__(self, train_data, **params):
        super().__init__(train_data, **params)

    def rank_candidates(self, q, candidates):
        return np.random.choice(len(candidates), len(candidates), False)

    def get_params(self):
        return dict()


class RankingGeneratorBestMatch(RankingGeneratorBase):
    def __init__(self, train_data, **params):
        super().__init__(train_data, **params)

    def __get_best_match(self, q):
        loci = self.get_loci(q)
        return self._locations[
            np.argsort(G(self._train_data[loci].values, q[loci].values))[-1]
        ]

    def rank_candidates(self, q, candidates):
        best_match_location = self.__get_best_match(q)
        return -D(candidates, best_match_location)

    def get_params(self):
        return dict()


class RankingGeneratorGGredy(RankingGeneratorBase):
    def __init__(self, train_data, **params):
        super().__init__(train_data, **params)

    def rank_candidates(self, q, candidates):
        loci = self.get_loci(q)
        g_similarity = G(self._train_data[loci].values, q[loci].values)
        loc_to_sim = dict()
        for index, loc in enumerate(self._locations):
            loc_to_sim[tuple(loc)] = np.max(
                [loc_to_sim.get(tuple(loc), 0),
                 g_similarity[index]]
            )

        return [loc_to_sim[tuple(c)] for c in candidates]

    def get_params(self):
        return dict()


class RankingGeneratorGDLinkage(RankingGeneratorBase):
    def __init__(self, train_data, **params):
        super().__init__(train_data, **params)

    def get_genotype_distances(self, q):
        loci = self.get_loci(q)
        return G_f(self._train_data[loci].values,
                   q[loci].values,
                   **self._params)

    def get_location_distances(self, candidates):
        return np.hstack([
            D_f(self._locations, candidate, **self._params).reshape(-1, 1)
            for candidate in candidates
        ])

    def rank_candidates(self, q, candidates):
        genotype_d = self.get_genotype_distances(q)
        location_d = self.get_location_distances(candidates)

        # Thresholds
        genotype_d = genotype_d * (genotype_d > self._params.get('g_t', 0))
        location_d = location_d * (location_d > self._params.get('d_t', 0))

        return np.dot(genotype_d, location_d)

    def get_params(self):
        return {
            'g_t': (0, 0.3),
            'd_t': (0, 0.3),
            'd_alpha': (0.5, 1.5),
            'g_mean': (0.2, 0.8),
            'g_std': (1, 10)
        }

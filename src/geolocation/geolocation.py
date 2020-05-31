import numpy as np

from geolocation.candidates_generation import (FrequentLocationsGenerator,
                                               GridGenerator)
from geolocation.candidates_ranking import (RankingGeneratorBestMatch,
                                            RankingGeneratorGDLinkage,
                                            RankingGeneratorGGredy,
                                            RankingGeneratorRandom)


class Geolocator:
    def __init__(self):
        self._train_data = None
        self._c_gen = None
        self._rank_gen = None

    def set_train_data(self, train_data):
        self._train_data = train_data
        return self

    def set_candidates_generator(self, generator, **params):
        if generator == 'grid':
            self._c_gen = GridGenerator(**params)
        else:
            self._c_gen = FrequentLocationsGenerator(
                train_data=self._train_data, **params)
        return self

    def set_ranking_generator(self, ranker, **params):
        if ranker == 'random':
            self._rank_gen = RankingGeneratorRandom(self._train_data, **params)
        if ranker == 'best_match':
            self._rank_gen = RankingGeneratorBestMatch(
                self._train_data, **params)
        if ranker == 'gdlink':
            self._rank_gen = RankingGeneratorGDLinkage(
                self._train_data, **params)
        if ranker == 'greedy':
            self._rank_gen = RankingGeneratorGGredy(
                self._train_data, **params)
        return self

    def locate(self, q, k=30):
        candidates = self._c_gen.generate_candidates()
        ranks = self._rank_gen.rank_candidates(q, candidates)
        return candidates[np.argsort(ranks)[::-1]][:k]

    def batch_locate(self, Q, k=30):
        return [self.locate(Q.iloc[i:i + 1], k) for i, _ in Q.iterrows()]


AVAILABLE_GEOLOCATORS = {
    'random': lambda train_data, cg_params, rg_params: (
        Geolocator()
        .set_train_data(train_data)
        .set_candidates_generator('freq', **cg_params)
        .set_ranking_generator('random', **rg_params)
    ),
    'best_match': lambda train_data, cg_params, rg_params: (
        Geolocator()
        .set_train_data(train_data)
        .set_candidates_generator('freq', **cg_params)
        .set_ranking_generator('best_match', **rg_params)
    ),
    'greedy': lambda train_data, cg_params, rg_params: (
        Geolocator()
        .set_train_data(train_data)
        .set_candidates_generator('freq', **cg_params)
        .set_ranking_generator('greedy', **rg_params)
    ),
    'gdlink': lambda train_data, cg_params, rg_params: (
        Geolocator()
        .set_train_data(train_data)
        .set_candidates_generator('freq', **cg_params)
        .set_ranking_generator('gdlink', **rg_params)
    )
}

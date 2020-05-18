import pandas as pd
from folium import Map
from folium.plugins import HeatMap

from dash_app.geolocation.config import (MAP_DEFAULT_START_PARAMS,
                                         MAX_CANDIDATES)
from geolocation.candidates_ranking import CandidatePointsGenerator
from geolocation.config import AVAILABLE_LOCI


def build_heat_map(data):
    heat_map = Map(**MAP_DEFAULT_START_PARAMS)
    HeatMap(data[['lat', 'long']], radius=10).add_to(heat_map)
    return heat_map


def sample_to_df(sample):
    return pd.DataFrame({
        # Meta data section
        **{k: [v] for k, v in sample.get('meta_data', dict()).items()},
        # Genotype
        **{k: [v] for k, v in sample.get('genotype', dict()).items()}
    })


def get_candidate_locations(train, sample, hyperparams):
    loci = [
        locus
        for locus in AVAILABLE_LOCI
        if locus in train.columns and locus in sample.columns
    ]
    cpg = CandidatePointsGenerator(train, loci)

    return pd.DataFrame(
        cpg.get_ranked_candidates(sample, hyperparams, MAX_CANDIDATES),
        columns=['lat', 'long', 'signal'])

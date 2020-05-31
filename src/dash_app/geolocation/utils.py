import pandas as pd

from dash_app.geolocation.config import MAP_DEFAULT_START_PARAMS
from folium import Circle, Map
from folium.plugins import HeatMap
from geolocation.geolocation import AVAILABLE_GEOLOCATORS


def build_heat_map(data, sample):
    heat_map = Map(**MAP_DEFAULT_START_PARAMS)
    HeatMap(data[['lat', 'long']], radius=10).add_to(heat_map)
    if 'lat' in sample.columns:
        heat_map.add_child(Circle(*sample[['lat', 'long']].values,
                                  radius=1.1e5))
    return heat_map


def sample_to_df(sample):
    return pd.DataFrame({
        # Meta data section
        **{k: [v] for k, v in sample.get('meta_data', dict()).items()},
        # Genotype
        **{k: [v] for k, v in sample.get('genotype', dict()).items()}
    })


def get_candidate_locations(train, sample, params):
    result_locations = (
        AVAILABLE_GEOLOCATORS[params['method']](train, {}, {})
        .batch_locate(sample, params['k'])
    )
    print(result_locations)
    return pd.DataFrame(
        result_locations[0],
        columns=['lat', 'long'])

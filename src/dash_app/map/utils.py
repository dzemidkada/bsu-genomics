import base64

import folium
import pandas as pd

from config import (MAP_DEFAULT_START_PARAMS, REGIONS_COLOR_MAPPING,
                    REGIONS_MAPPING)


def read_image(path):
    return base64.b64encode(open(path, 'rb').read()).decode('ascii')


def filter_by_region(data, region):
    filtered_data = (
        data
        if region == 'all' else
        data.query(f'region == "{REGIONS_MAPPING[region]}"')
    )
    return filtered_data


def build_map(filtered_data):
    m = folium.Map(**MAP_DEFAULT_START_PARAMS)

    for _, row in filtered_data.iterrows():
        if not row.valid:
            return
        marker = folium.Marker(
            [row.lat, row.long],
            popup=f'{row.id}',
            icon=folium.Icon(
                color=REGIONS_COLOR_MAPPING[row.region],
                icon='ok-sign')
        )
        m.add_child(marker)

    return m


def update_source(source_df, affected_df):
    if 'region' in affected_df.columns:
        regions = affected_df.region.unique().tolist()
        if len(regions) > 1:
            source_df = affected_df
        else:
            source_df = pd.concat([
                source_df[~source_df.region.isin(regions)],
                affected_df
            ]).reset_index(drop=True)
    return source_df


def dummy_df():
    return pd.DataFrame({'col': ['value']})

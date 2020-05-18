import os
import traceback

import dash
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output

from dash_app.geolocation.config import TRAIN_DATA_TABLE
from dash_app.geolocation.layout import genotype_viewer_layout, main_layout
from dash_app.geolocation.utils import build_heat_map, get_candidate_locations

os.chdir(os.environ['PROJECT_ROOT'])


def init_dependecies():
    global TRAIN_DATA, SAMPLE
    TRAIN_DATA = pd.read_excel(TRAIN_DATA_TABLE)
    SAMPLE = None


init_dependecies()


app = dash.Dash(__name__)

app.layout = main_layout()


@app.callback([Output('genotype-table-div', 'children'),
               Output('heat-map-gen-button', 'style')],
              [Input('genotype-upload', 'contents'),
               Input('genotype-upload', 'filename')])
def upload_genotype(contents, filename):
    global SAMPLE
    SAMPLE = None

    if contents is None:
        return html.Div(), dict(display='none')

    SAMPLE, result_layout = genotype_viewer_layout(contents, filename)
    return result_layout, dict()


@app.callback(Output('heat-map-div', 'children'),
              [Input('heat-map-gen-button', 'n_clicks'),
               Input('grid_size', 'value'),
               Input('distance_sensitivity', 'value'),
               Input('distance_threshold', 'value'),
               Input('similarity_threshold', 'value')])
def create_heat_map(_,
                    grid_size, distance_sensitivity,
                    distance_threshold, similarity_threshold):
    global SAMPLE

    if SAMPLE is None:
        return html.Div()

    try:
        heat_map = build_heat_map(
            get_candidate_locations(
                TRAIN_DATA, SAMPLE,
                dict(grid_size=grid_size,
                     distance_sensitivity=distance_sensitivity,
                     distance_threshold=distance_threshold,
                     similarity_threshold=similarity_threshold)))
    except:
        traceback.print_exc()

    return html.Iframe(
        id='folium-interative-map',
        srcDoc=heat_map._repr_html_()
    )


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

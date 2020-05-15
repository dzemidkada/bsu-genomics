import os

import dash
import pandas as pd
from dash.dependencies import Input, Output

from config import SURVEYS_DATA_PATH
from layout import (data_overview_layout, main_layout, new_placeholder,
                    samples_map_layout)
from utils import filter_by_region, update_source

os.chdir(os.environ['PROJECT_ROOT'])


def init_dependecies():
    global SOURCE_DATA_TABLE
    SOURCE_DATA_TABLE = pd.read_csv(SURVEYS_DATA_PATH).sample(10)
    SOURCE_DATA_TABLE['valid'] = True


init_dependecies()


app = dash.Dash(__name__)

app.layout = main_layout()


@app.callback(Output('data-table-div', 'children'),
              [Input('region-radio', 'value')])
def render_data_table_content(region):
    global SOURCE_DATA_TABLE

    filtered_data = filter_by_region(SOURCE_DATA_TABLE, region)

    return data_overview_layout(filtered_data)


@app.callback(Output('map-visualization-div', 'children'),
              [Input('region-radio', 'value')])
def render_map_visualization_content(region):
    global SOURCE_DATA_TABLE

    filtered_data = filter_by_region(SOURCE_DATA_TABLE, region)

    return samples_map_layout(filtered_data)


@app.callback(
    Output('placeholder', 'children'),
    [Input('data_table', 'data'),
     Input('data_table', 'columns')]
)
def save_data_table_changes(rows, columns):
    global SOURCE_DATA_TABLE

    affected_subset = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    SOURCE_DATA_TABLE = update_source(SOURCE_DATA_TABLE, affected_subset)
    return new_placeholder()


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

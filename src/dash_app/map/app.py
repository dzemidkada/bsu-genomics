import json
import base64
import os
os.chdir(os.environ['PROJECT_ROOT'])

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bio
import dash_table
import pandas as pd


def init_dependecies():
    # Tab 1
    global data_table, view_data_table
    source_data_table = pd.read_csv('data/apps/map/survey_data_v1.csv')
    view_data_table = source_data_table.copy()

    global regions_map, regions_map_encoded
    regions_map = 'data/apps/map/belarus_regions.png'
    regions_map_encoded = base64.b64encode(open(regions_map, 'rb').read()).decode('ascii')


init_dependecies()


app = dash.Dash(__name__)

app.layout = html.Div(id='map-body', className='app-body', children=[
    html.Div([
        html.Div(className='h-text', children=[
            html.H1('Geo visualization')
        ]),
        dcc.Tabs(id='map-app-tabs', value='data-overview-tab', children=[
            dcc.Tab(
                label='Survey data overview',
                value='data-overview-tab',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Geo visualization',
                value='map-tab',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
        ]),
        html.Div(id='tab-content')
    ])
])


def envelope_dataframe(df):
    def _data_table_style():
        return {
            'page_action': 'none',
            'fixed_rows': {'headers': True},
            'style_table': {
                "width": 'auto', 'height': '600px', 'overflowX': 'auto', 'overflowY': 'auto'
            },
            'style_cell': {
                'whiteSpace': 'normal', 'height': 'auto',
                },
            'style_data': {
                'whiteSpace': 'normal', 'height': 'auto'
            },
            'filter_action': "native",
            'sort_action': "native",
            'sort_mode': "multi"
        }

    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        #**_data_table_style()
    )

def data_overview_layout():
    return html.Div(className='h-text', children=[
        html.H3('Survey Data'),

        html.Div(
            id='surveys-data-table',
            children=[
                envelope_dataframe(view_data_table)
            ]
        )
    ])

def generate_folium_map():
    return html.P('kek')
    # html.Iframe(
    #     id='folium-interative-map',
    #     srcDoc=open('data/apps/map/survey_data_v1.html', 'r').read()
    # )

def samples_map():
    return html.Div([
        html.Div(className='h-text', children=[
            html.H3('BelPop2018 + Autosomal2020'),
        ]),
        html.Div(
            className='row',
            children=[
                html.Div(className='two-thirds column', children=[
                    generate_folium_map()
                ]),
                # Reference image
                html.Img(
                    className='one-third column',
                    id='static-regions-map',
                    src='data:image/png;base64,{}'.format(regions_map_encoded)
                )
            ]
        )
    ])

@app.callback(Output('tab-content', 'children'),
              [Input('map-app-tabs', 'value')])
def render_content(tab):
    if tab == 'data-overview-tab':
        return data_overview_layout()
    if tab == 'map-tab':
        return samples_map()


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

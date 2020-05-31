import base64
import io
import json

import dash_core_components as dcc
import dash_html_components as html
from dash_app.geolocation.utils import sample_to_df
from dash_app.map.layout import create_data_table


def main_layout():
    return html.Div(id='geolocation-body', className='app-body', children=[
        html.Div([
            html.Div(className='h-text', children=[
                html.H1('Geolocation utility')
            ]),
            html.Hr(),
            dcc.Upload(
                html.Button('Please uploade a genotype (JSON format only)'),
                id='genotype-upload',
                multiple=False
            ),
            html.Div(id='genotype-table-div'),
            html.Hr(),
            html.Div(
                className='row',
                children=[
                    html.Div(className='one-third column', children=[
                        settings_layout(),
                        html.Button('Locate', id='heat-map-gen-button',
                                    n_clicks=0, style=dict(display='none'))
                    ]),
                    html.Div(className='two-thirds column', children=[
                        html.Div(id='heat-map-div')
                    ]),
                ]
            ),
            html.P(id='placeholder')
        ])
    ])


def settings_layout():
    return html.Div(children=[
        html.H5('Settings'),
        html.Div(children=[
            html.P('Location method'),
            dcc.RadioItems(
                id='method',
                options=[
                    {'label': 'Random', 'value': 'random'},
                    {'label': 'Best Match', 'value': 'best_match'},
                    {'label': 'Greedy', 'value': 'greedy'},
                    {'label': 'GDlink', 'value': 'gdlink'}
                ],
                value='random'
            )
        ]),
        html.Div(children=[
            html.P('# Candidates'),
            dcc.Slider(
                id='candidates_k',
                min=1, max=50, step=1, value=10,
                marks={5: '5', 10: '10', 20: '20'}
            ),
        ]),
    ])


def genotype_viewer_layout(contents, _):
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        sample = json.load(io.StringIO(decoded.decode('utf-8')))

        sample_df = sample_to_df(sample)
    except:
        return html.P('Something went wrong. '
                      'Please the file format, cannot deserialize json.')

    return sample_df, html.Div(className='h-text', children=[
        html.H3('Genotype'),
        html.Div(
            id='genotype-data-table',
            children=[
                create_data_table(sample_df,
                                  editable=False, table_height=100,
                                  fixed_headers=False)
            ]
        )
    ])

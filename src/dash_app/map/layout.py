import dash_core_components as dcc
import dash_html_components as html
import dash_table

from dash_app.map.config import REGIONS_MAP_REF, REGIONS_MAPPING
from dash_app.map.utils import build_map, dummy_df, read_image


def main_layout():
    return html.Div(id='map-body', className='app-body', children=[
        html.Div([
            html.Div(className='h-text', children=[
                html.H1('Geo visualization')
            ]),
            html.Div(children=[
                html.P('Please, pick ethnographic region of interest'),
                dcc.RadioItems(
                    id='region-radio',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        *[
                            {'label': region, 'value': region}
                            for region in REGIONS_MAPPING
                        ]
                    ],
                    value='all',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                html.Div(className='h-text', children=[
                    html.H3('Samples geolocation (BelPop2018 + Autosomal2020)'),
                ]),
                html.Div(
                    className='row',
                    children=[
                        html.Div(className='two-thirds column', children=[
                            html.Div(id='map-visualization-div')
                        ]),
                        ethnogeographic_regions_image()
                    ]
                )
            ]),
            html.Div(id='data-table-div', children=[
                data_overview_layout(dummy_df())
            ]),
            html.P(id='placeholder')
        ])
    ])


def ethnogeographic_regions_image():
    return html.Img(
        className='one-third column',
        id='static-regions-map',
        src='data:image/png;base64,{}'.format(read_image(REGIONS_MAP_REF))
    )


def create_data_table(filtered_data, editable=True, table_height=600, fixed_headers=True):
    def _data_table_style():
        return {
            'page_action': 'none',
            'fixed_rows': {'headers': fixed_headers},
            'style_table': {
                'height': f'{table_height}px',
                'overflowX': 'auto', 'overflowY': 'auto'
            },
            'style_cell': {
                'whiteSpace': 'normal', 'height': 'auto', 'padding': '3px',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            'style_data': {
                'whiteSpace': 'normal', 'height': 'auto'
            },
            # 'filter_action': "native",
            'sort_action': "native",
            'sort_mode': "multi",
            'style_header': {
                'backgroundColor': 'white',
                'fontWeight': 'bold',
                'width': 'auto'
            }
        }

    return dash_table.DataTable(
        id='data_table',
        columns=[{"name": i, "id": i} for i in filtered_data.columns],
        data=filtered_data.to_dict('records'),
        editable=editable,
        export_format='xlsx',
        export_headers='display',
        **_data_table_style()
    )


def data_overview_layout(filtered_data):
    return html.Div(className='h-text', children=[
        html.H3('Survey Data'),
        html.P(f'{filtered_data.shape[0]} samples available'),
        html.Div(
            id='surveys-data-table',
            children=[
                create_data_table(filtered_data)
            ]
        )
    ])


def samples_map_layout(filtered_data):
    result_map = build_map(filtered_data)
    return html.Iframe(
        id='folium-interative-map',
        srcDoc=result_map._repr_html_()
    )


def new_placeholder():
    return html.P()

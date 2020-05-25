import dash_core_components as dcc
import dash_html_components as html
from dash_bio import SequenceViewer

from dash_app.map.layout import create_data_table
from dash_app.str_annotator.config import (GENERATOR_READS_TO_USE,
                                           VIEWER_READS_TO_SHOW)
from dash_app.str_annotator.utils import (generate_potential_patterns,
                                          retrieve_reads)
from str_finder.utils import get_most_common_reads


def main_layout():
    return html.Div(id='str-annotator-body', className='app-body', children=[
        html.Div([
            html.Div(className='h-text', children=[
                html.H1('STR annotation')
            ]),
            dcc.Tabs(id='utils-tabs', value='seq-viewer', children=[
                dcc.Tab(label='Sequence viewer',
                        value='seq-viewer-tab',
                        children=[
                            html.Div(children=[
                                dcc.Upload(
                                    html.Button('Upload File'),
                                    id='tab-1-upload',
                                    multiple=False
                                ),
                                html.Hr()
                            ]),
                            html.Div(id='read-results-tab-1')
                        ]),
                dcc.Tab(label='Repeat patterns generation',
                        value='rep-gen-tab',
                        children=[
                            html.Div(children=[
                                dcc.Upload(
                                    html.Button('Upload File'),
                                    id='tab-2-upload',
                                    multiple=False
                                ),
                                html.Hr()
                            ]),
                            html.Div(id='read-results-tab-2')
                        ])
            ]),
            html.Div(id='utils-tabs-content')
        ])
    ])


def viewer_tab_layout(contents, filename):
    reads = retrieve_reads(contents)
    most_common_reads = get_most_common_reads(reads, VIEWER_READS_TO_SHOW)
    result_div_children = [
        html.H5(filename),
        html.P(f'{len(reads)} reads within the file'),
    ]
    if most_common_reads:
        result_div_children.append(html.H5('The most common reads:'))
        for i, (read, occ) in enumerate(most_common_reads):
            result_div_children.append(
                html.Div([
                    html.Hr(),
                    SequenceViewer(
                        id=f'sequence-viewer-mcr-{i}',
                        sequence=read,
                        title=f'Read #{i+1} ({occ} occurences)',
                        badge=False,
                        toolbar=True
                    )
                ])
            )

    return html.Div(result_div_children)


def pattern_generation_tab_layout(contents, filename):
    reads = retrieve_reads(contents)
    most_common_reads = get_most_common_reads(reads, GENERATOR_READS_TO_USE)
    result_div_children = [
        html.H5(filename),
        html.P(f'{len(reads)} reads within the file'),
    ]
    if most_common_reads:
        potential_patterns_df = generate_potential_patterns(most_common_reads)

        result_div_children.extend([
            html.H3('Potential patterns'),
            create_data_table(potential_patterns_df)
        ])

    else:
        result_div_children.append(
            'No patterns were found. Please, check the file')

    return html.Div(result_div_children)

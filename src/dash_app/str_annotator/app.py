import os

import dash
import dash_html_components as html
from dash.dependencies import Input, Output

from dash_app.str_annotator.layout import (main_layout,
                                           pattern_generation_tab_layout,
                                           viewer_tab_layout)

os.chdir(os.environ['PROJECT_ROOT'])


def init_dependecies():
    pass


init_dependecies()


app = dash.Dash(__name__)

app.layout = main_layout()


@app.callback(Output('read-results-tab-1', 'children'),
              [Input('tab-1-upload', 'contents'),
               Input('tab-1-upload', 'filename')])
def update_tab1_output(contents, filename):
    if contents is None:
        return html.Div()

    return viewer_tab_layout(contents, filename)


@app.callback(Output('read-results-tab-2', 'children'),
              [Input('tab-2-upload', 'contents'),
               Input('tab-2-upload', 'filename')])
def update_tab2_output(contents, filename):
    if contents is None:
        return html.Div()

    return pattern_generation_tab_layout(contents, filename)


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

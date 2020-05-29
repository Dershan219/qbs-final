import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import random
import plotly.graph_objs as go
from collections import deque

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Stock Prices"),
    dcc.Input(id='input', value='TSLA', type='text'),
    html.Div(id='output-graph')
    ])

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
)

def update_graph(input_data):
    start = datetime(2018, 1, 1)
    end = datetime.now()
    df = web.DataReader(input_data, 'yahoo', start, end)
    try:
        return dcc.Graph(id='stock-graph',
                        figure={
                            'data':[
                                {'x':df.index, 'y':df.Close, 'type':'line', 'name':input_data}
                                ],
                                'layout':{
                                    'title':input_data
                                    }
                                    })
    except:
        pass

if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import sqlite3
import time
import os

os.chdir('C:/Users/Andy/Desktop/NTU Courses/Quantitative Business Science/Project')

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H2("Live Twitter Sentiment"),
    dcc.Input(id='keyword', value='Trump', type='text'),
    dcc.Graph(id='live-graph', animate=False),
    dcc.Interval(id='graph-update', interval=1*1000)
    ])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('keyword', 'value'),
    Input('graph-update', 'n_intervals')]
)

def update_graph(keyword, n):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql('SELECT * FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 200', conn, params=('%' + keyword + '%', ))
    df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)

    X = df.index
    Y = df.sentiment_smoothed

    data = go.Scatter(
        x=X,
        y=Y,
        name='Scatter',
        mode='lines+markers'
    )

    return {'data':[data], 'layout':go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                              yaxis=dict(range=[min(Y), max(Y)]),
                                              title='Term: {}'.format(keyword))}

if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3
import time
import os
import re

os.chdir('C://Users/Andy/Desktop/NTU Courses/Quantitative Business Science/Project/twitter-sentiment')

# global settings-----------------------------------------------------------

app_colors = {
    'background':'#272B30',
    'text':'#AAAAAA',
    'plot1':'#FFFFFF',
    'plot2':'#B54057',
    'plot3':'#5C5C5C'
}

sentiment_colors = {0:"#FFE6AC",
                    0.5:"#D0F2DF",
                    1:"#9CEC5B"}

LOGO = "https://help.twitter.com/content/dam/help-twitter/brand/logo.png"
THEME = dbc.themes.SLATE
REFRESH = 1000 # ms

app = dash.Dash(__name__,
                external_stylesheets=[THEME])

#%%
# layout settings-----------------------------------------------------------
header = dbc.Navbar(
    [
        dbc.Col(html.Img(src=LOGO, height="30px"), md=0.5),
        dbc.Col(
            dbc.NavbarBrand(
                "Twitter Sentiment",
                href="https://github.com/Dershan219/twitter-sentiment"),
                style={'font-weight':'bold'},
                md=2
                ),
        dbc.Col(
            dbc.Input(
                id='keyword', value='Trump',
                type='text', placeholder="Search Term",
                style={'height':'36px', 'margin-top':'0px'}),
                md=9
                ),
        dbc.Col(
            dbc.Button(
                "Search", color="primary",
                style={'height':'36px', 'margin-top':'0px', 'padding':'0rem 1rem'}
                ),
            md=0.5
            )
    ],
    color="dark",
    dark=True,
)

body = dbc.Container(
    [
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Button('Loading related terms...', id='live-term'), width=6),
                dcc.Interval(id='term-update', interval=10*REFRESH)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='live-graph', animate=False), width=12),
                dcc.Interval(id='graph-update', interval=REFRESH)
            ],
            justify='center'
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        "Most Negative Tweets",
                        style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                    width=6),
                dbc.Col(
                    html.H5(
                        "P/N Ratio",
                        style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                    width=6),               
            ],
            justify='between'
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Div(id='live-table'), width=6),
                dcc.Interval(id='table-update', interval=REFRESH),
                dbc.Col(dcc.Graph(id='live-pie', animate=False), width=6),
                dcc.Interval(id='pie-update', interval=10*REFRESH),
            ],
            justify='between'
        )
    ]
)

footer = dbc.Container(
    [
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H6(
                    "QBS Group 5",
                    style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                    ), width = 6
                )
            ],
            justify = 'center'
        )
    ])


#%%
# sentiment graph-----------------------------------------------------------

# def generate_terms(df):


# @app.callback(
#     Output('live-term', 'children'),
#     [Input('keyword', 'value'),
#      Input('term-update', 'n_intervals')]
# )

# def update_terms(keyword, n):


@app.callback(
    Output('live-graph', 'figure'),
    [Input('keyword', 'value'),
     Input('graph-update', 'n_intervals')]
)

def update_graph(keyword, n):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        'SELECT * FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 2000',
        conn, params=('%' + keyword + '%', ))
    
    df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
    df['date'] = pd.to_datetime((df['time']/1000).astype(int), unit='s')
    df.set_index('date', inplace=True)
    
    df = df.resample('10s').agg({"id":'size',"sentiment_smoothed":'mean'})
    df['id'].fillna(0, inplace=True)
    df['sentiment_smoothed'].fillna(method='ffill', inplace=True)

    X = df.index
    Y1 = df.sentiment_smoothed.values.round(2)
    Y2 = df.id.values

    data = go.Scatter(
        x=X,
        y=Y1,
        name='Scatter',
        mode='lines',
        line=dict(color=app_colors["plot1"], width=3),
        yaxis='y')

    data2 = go.Bar(
        x=X,
        y=Y2,
        name='Bar',
        marker=dict(color=app_colors["text"], opacity=0.8),
        yaxis='y2')

    layout = go.Layout(
        xaxis=dict(
            range=[min(X), max(X)],
            color=app_colors["text"],
            showspikes=True),
        yaxis=dict(
            range=[min(Y1),max(Y1)],
            title='sentiment', side='left', overlaying='y2',
            color=app_colors["text"],
            showgrid=False, showspikes=True),
        yaxis2=dict(
            range=[min(Y2),max(Y2*4)],
            title='volume', side='right',
            color=app_colors["text"],
            showgrid=False, showspikes=True),
        showlegend=False,
        plot_bgcolor=app_colors["background"],
        paper_bgcolor=app_colors["background"],
        title=dict(
            text='Keyword: {}'.format(keyword),
            font=dict(color=app_colors["plot1"]),
            x=0.5,
            y=0.9,
            xanchor='center',
            yanchor='top'),
        margin=go.layout.Margin(
            l=50,
            r=50,
            pad=15
        ))

    return go.Figure(
        data=[data, data2], 
        layout=layout)

#%%
#most negative tweets-------------------------------------------------------
def generate_table(df):
    return dash_table.DataTable(
        id='live-table',
        columns=[{"name": i, "id": i, "selectable":True} for i in df.columns],
        data=df.to_dict('records'),
        filter_action="native",
        sort_action="native",
        style_header={
            'backgroundColor':app_colors["plot3"],
            'color':app_colors["plot1"],
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border-bottom': '2px solid {}'.format(app_colors["background"])},
        style_cell={
            'backgroundColor':app_colors["text"],
            'color':app_colors["background"], 'height': 'auto',
            'textAlign': 'center',
            'border': '1px solid {}'.format(app_colors["background"])},
        style_cell_conditional=[
            {'if': {'column_id': 'tweet'},
            'width': '480px',
            'minWidth': '480px',
            'maxWidth': '480px',
            'whiteSpace': 'normal',
            'textAlign': 'left'
            }],
        page_size=5,
        css=[
            {'selector': '.previous-page, .next-page, .last-page, .first-page',
            'rule': 'background-color: #272B30;'}
            ]
        )

@app.callback(
    Output('live-table', 'children'),
    [Input('keyword', 'value'),
     Input('table-update', 'n_intervals')]
)

def update_negative_tweets(keyword, n):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        'SELECT tweet, sentiment FROM tweets WHERE tweet LIKE ? ORDER BY sentiment LIMIT 30',
        conn, params=('%' + keyword + '%', )
    )
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"RT|@\S+|\Whttps:\S+|\Whttp:\S+|\.\.\.",' ', x))
    df['sentiment'] = df['sentiment'].apply(lambda x: re.sub("(?<=....)(.*?)(?=e.+)", '', str(x))).astype(float)
    return generate_table(df)

#%%
# P/N Ratio
@app.callback(
    Output('live-pie', 'figure'),
    [Input('keyword', 'value'),
     Input('pie-update', 'n_intervals')]
)

def update_pie(keyword, n):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        'SELECT * FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 2000',
        conn, params=('%' + keyword + '%', ))

    df['sentiment_binary'] = df['sentiment'].apply(lambda x: 'negative' if x<0.5 else 'positive')

    sentiment = df['sentiment_binary'].value_counts().values

    data = go.Pie(
        labels=['positive', 'negative'],
        values=sentiment,
        textinfo='label+percent',
        marker=dict(colors=[app_colors['text'], app_colors['plot2']])
    )

    layout = go.Layout(
        showlegend=False,
        plot_bgcolor=app_colors["background"],
        paper_bgcolor=app_colors["background"],
        height=280,
        margin=go.layout.Margin(
            t=0,
            b=0
        )
    )

    return go.Figure(
        data=data,
        layout=layout
    )

app.layout = html.Div([header, body, footer])

if __name__ == '__main__':
    app.run_server(debug=True)
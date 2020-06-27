import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sqlite3
import time
import os
import re
import pickle
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from PIL import Image, ImageOps
import requests

os.chdir('C://Users/Andy/Desktop/NTU Courses/Quantitative Business Science/Project/twitter-sentiment')

# global settings-----------------------------------------------------------

app_colors = {
    'background':'#272B30',
    'text':'#AAAAAA',
    'plot1':'#FFFFFF',
    'plot2':'#B54057',
    'plot3':'#5C5C5C',
    'plot4':'#32383E',
    'plot5':'#00acee'
}

LOGO = "https://help.twitter.com/content/dam/help-twitter/brand/logo.png"
THEME = dbc.themes.SLATE
REFRESH = 1000 # ms
DASH_CSS = "./dash_style.css"

# preparing LSTM model------------------------------------------------------
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_coef(y_true, y_pred, smooth=1e-12):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

model = load_model(
    'train_model_LSTM.h5',
    custom_objects={
        'jaccard_distance_loss':jaccard_distance_loss,
        'jaccard_coef':jaccard_coef})

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stop_words = set(stopwords.words('english'))
stop_words.update(['rt', 'amp'])
stop_words.remove('not')

def preprocess(text):
    review=re.sub(r"RT |@\S+ |https:\S+|http:\S+|[^a-zA-Z0-9' ]",' ',text)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stop_words]
    review_tokenized=pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=300)
    return review, review_tokenized

app = dash.Dash(__name__, external_stylesheets=[THEME, DASH_CSS])

#%%
# layout settings-----------------------------------------------------------
header = dbc.Navbar(
    [
        dbc.Col(html.Img(src=LOGO, height="36px"), md=0.5),
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
                style={'height':'36px', 'margin-top':'0px'}
            ),
            md=9
        ),
        dbc.Col(
            dbc.Button(
                "Search", id="term-search", n_clicks=0, color="primary",
                style={'height':'36px', 'margin-top':'0px', 'padding':'0rem 1rem'}
                ),
            md=0.5
        ),
        html.Div(children=[""],id='input-temp', style={'display': 'none'})
    ],
    color="dark",
    dark=True,
)

tab1 = dbc.Card(
    dbc.CardBody(
        [
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(width=1),
                    dbc.Col(
                        html.H6(
                            "Related Terms: ",
                            style={
                                'color':'{}'.format(app_colors['plot5']), 'font-weight':'bold',
                                'margin-right':'0.3rem', 'padding':'0.1rem', 'font-size':'1rem'}
                        ),
                        width=1.5
                    ),
                    dbc.Col(
                        dbc.Button(
                            [dbc.Spinner(size="sm"),' Loading...'],
                            id='live-term1',
                            color='rgba(0,0,0,0)',
                            style={
                                'line-height':'1.2', 'padding':'0rem 0.3rem',
                                'vertical-align':'top', 'border-color':'rgba(0,0,0,0)',
                                'font-size':'1rem',
                                'color':'{}'.format(app_colors['plot1']), 'text-shadow':'none'
                            },
                            n_clicks=0      
                        ),
                        width=0.5
                    ),
                    dbc.Col(
                        dbc.Button(
                            '', id='live-term2',
                            color='rgba(0,0,0,0)',
                            style={
                                'line-height':'1.2', 'padding':'0rem 0.3rem',
                                'vertical-align':'top', 'border-color':'rgba(0,0,0,0)',
                                'font-size':'1rem',
                                'color':'{}'.format(app_colors['plot1']), 'text-shadow':'none'
                            },
                            n_clicks=0
                        ),
                        width=0.5
                    ),
                    dbc.Col(
                        dbc.Button(
                            '', id='live-term3',
                            color='rgba(0,0,0,0)',
                            style={
                                'line-height':'1.2', 'padding':'0rem 0.3rem',
                                'vertical-align':'top', 'border-color':'rgba(0,0,0,0)',
                                'font-size':'1rem',
                                'color':'{}'.format(app_colors['plot1']), 'text-shadow':'none'
                            },
                            n_clicks=0
                        ),
                        width=0.5
                    ),
                    dbc.Col(
                        dbc.Button(
                            '', id='live-term4',
                            color='rgba(0,0,0,0)',
                            style={
                                'line-height':'1.2', 'padding':'0rem 0.3rem',
                                'vertical-align':'top', 'border-color':'rgba(0,0,0,0)',
                                'font-size':'1rem',
                                'color':'{}'.format(app_colors['plot1']), 'text-shadow':'none'
                            },
                            n_clicks=0
                        ),
                        width=0.5
                    ),
                    dbc.Col(
                        dbc.Button(
                            '', id='live-term5',
                            color='rgba(0,0,0,0)',
                            style={
                                'line-height':'1.2', 'padding':'0rem 0.3rem',
                                'vertical-align':'top', 'border-color':'rgba(0,0,0,0)',
                                'font-size':'1rem',
                                'color':'{}'.format(app_colors['plot1']), 'text-shadow':'none'
                            },
                            n_clicks=0
                        ),
                        width=0.5
                    ),
                    dcc.Interval(id='term-update', interval=10*REFRESH)
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id='live-graph', animate=False
                        ),
                        width=12
                    ),
                    dcc.Interval(id='graph-update', interval=REFRESH)
                ],
                justify='center'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        html.H5(
                            "Word Cloud",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width=6
                    ),
                    dbc.Col(
                        html.H5(
                            "P/N Ratio",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width=6
                    ),               
                ],
                justify='between'
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([html.Img(id="image-wc")]),
                        width=6
                    ),
                    dcc.Interval(id='wc-update', interval=10*REFRESH),
                    dbc.Col(
                        dcc.Graph(id='live-pie', animate=False),
                        width=6
                    ),
                    dcc.Interval(id='pie-update', interval=10*REFRESH),
                ],
                justify='between'
            )
        ]   
    )
)

tab2 = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(width=1),
                    dbc.Col(
                        html.H5(
                            "Most Negative Tweets",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width=5
                    ),
                    dcc.Interval(id='neg-table-update', interval=30*REFRESH),
                                        dbc.Col(
                        html.H5(
                            "Most Positive Tweets",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width=5
                    )
                ],
                justify='start'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(width=1),
                    dbc.Col(
                        html.Div(id='neg-live-table'),
                        width=5
                    )
                ]
            )
        ]
    )
)

tab3 = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Textarea(
                            id='model-input', value='Test Out Our Model!',
                            placeholder="Input Text",
                            style={'height':'200px', 'margin-top':'0px'}
                        ),
                        width = 6
                    )
                ],
                justify = 'center'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Button(
                        'Submit', id='model-submit', n_clicks=0, color="primary",
                        style={'height':'36px', 'margin-top':'0px', 'padding':'0rem 1rem'}
                    )
                ],
                justify = 'center'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        html.H6(
                            "Score: ",
                            style={
                                'text-align':'right', 'font-size':'1rem',
                                'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width = 1
                    ),
                    dbc.Col(
                        html.Div(
                            children=[""], id='model-output',
                            style={
                                'text-align':'left', 'margin-left':'0.4rem',
                                'font-size':'1rem', 'line-height':'1.3',
                                'color':'{}'.format(app_colors['plot1'])}),
                        width = 1
                    )
                ],
                justify = 'center',
                no_gutters=True
            )
        ]
    )
)

tab4 = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H4(
                            "About the Dashboard",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width = 6
                    )
                ],
                justify = 'center'
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        html.H5(
                            "Training the Model",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width = 4
                    ),
                    dbc.Col(
                        html.H5(
                            "Streaming from Twitter",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width = 4
                    ),
                    dbc.Col(
                        html.H5(
                            "Building the Dashboard",
                            style={'text-align':'center', 'color':'{}'.format(app_colors['plot1'])}
                        ),
                        width = 4
                    )
                ]
            )
        ]
    )
)

body = dbc.Container(
    [
        html.Br(),
        dbc.Tabs(
            [
                dbc.Tab(tab1, label="Overview", tab_style={'margin-left':'auto'}),
                dbc.Tab(tab2, label="Dig Deeper!"),
                dbc.Tab(tab3, label="Our Model"),
                dbc.Tab(tab4, label="About")
            ]
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
                    ),
                    width = 6
                )
            ],
            justify = 'center'
        )
    ])

#%%
# related terms-------------------------------------------------------------
def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens: stems.append(PorterStemmer().stem(item))
    return stems

@app.callback(Output('input-temp', 'children'),
              [Input('term-search', 'n_clicks')],
              [State('keyword', 'value')])
def update_temp(click, keyword):
    return keyword

@app.callback(
    [Output('live-term1', 'children'),
    Output('live-term2', 'children'),
    Output('live-term3', 'children'),
    Output('live-term4', 'children'),
    Output('live-term5', 'children')],
    [Input('term-search', 'n_clicks'),
    Input('term-update', 'n_intervals'),
    Input('input-temp', 'children')]
)
def update_terms(click, n, keyword):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
    "SELECT tweet, keywords FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 2000",
    conn, params=('%' + keyword + '%', ))

    text = []
    for kw in df.keywords:
        seq = kw.split(',')
        seq = ' '.join(seq)
        text.append(seq)

    # word tokenize and stem
    text = [" ".join(tokenize(txt.lower())) for txt in text]
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.05)
    matrix = vectorizer.fit_transform(text).todense()
    # transform the matrix to a pandas df
    matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
    # sum over each document (axis=0)
    top_words = matrix.sum(axis=0).sort_values(ascending=False)

    imp_kw = pd.DataFrame(top_words.items(), columns=['word', 'freq'])
    rm_kw = [keyword.lower(), "not"]
    imp_kw = imp_kw[~ imp_kw['word'].str.contains('|'.join(rm_kw))].reset_index()
    del imp_kw['index']
    return imp_kw.word[0], imp_kw.word[1], imp_kw.word[2], imp_kw.word[3], imp_kw.word[4]

# change input into related terms
click1 = 0
click2 = 0
click3 = 0
click4 = 0
click5 = 0

@app.callback(
    Output('keyword', 'value'),
    [Input('live-term1', 'n_clicks'),
    Input('live-term2', 'n_clicks'),
    Input('live-term3', 'n_clicks'),
    Input('live-term4', 'n_clicks'),
    Input('live-term5', 'n_clicks')],
    [State('live-term1', 'children'),
    State('live-term2', 'children'),
    State('live-term3', 'children'),
    State('live-term4', 'children'),
    State('live-term5', 'children')]
)
def update_search(c1, c2, c3, c4, c5, kw1, kw2, kw3, kw4, kw5):
    global click1
    global click2
    global click3
    global click4
    global click5

    if c1!=click1:
        click1 +=1
        return kw1
    elif c2!=click2:
        click2 +=1
        return kw2
    elif c3!=click3:
        click3 +=1
        return kw3
    elif c4!=click4:
        click4 +=1
        return kw4
    elif c5!=click5:
        click5 +=1
        return kw5
    else:
        return "Trump"

#%%
# sentiment graph-----------------------------------------------------------
@app.callback(
    Output('live-graph', 'figure'),
    [Input('term-search', 'n_clicks'),
    Input('graph-update', 'n_intervals'),
    Input('input-temp', 'children')]
)
def update_graph(click, n, keyword):
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
        hovertemplate = "<b>%{x}<br>Sentiment Score: %{y}<extra></extra>",
        yaxis='y')

    data2 = go.Bar(
        x=X,
        y=Y2,
        name='Bar',
        marker=dict(
            color=app_colors["text"], opacity=0.8,
            line=dict(width=0)),
        hovertemplate = "<b>%{x}<br># of Tweets: %{y}<extra></extra>",
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
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='Keyword: {}'.format(keyword),
            font=dict(color=app_colors["plot1"]),
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'),
        margin=go.layout.Margin(
            t=40,
            l=50,
            r=50,
            pad=15
        ))

    return go.Figure(
        data=[data, data2], 
        layout=layout)

#%%
# P/N Ratio-----------------------------------------------------------------
@app.callback(
    Output('live-pie', 'figure'),
    [Input('term-search', 'n_clicks'),
    Input('pie-update', 'n_intervals'),
    Input('input-temp', 'children')]
)
def update_pie(click, n, keyword):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        'SELECT * FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 2000',
        conn, params=('%' + keyword + '%', ))

    df['sentiment_binary'] = df['sentiment'].apply(lambda x: 'negative' if x<0.5 else 'positive')

    sentiment = df['sentiment_binary'].value_counts().values

    data = go.Pie(
        labels=['positive', 'negative'],
        values=sentiment,
        texttemplate = "<b>%{label} <br>%{percent}",
        hovertemplate = "<b>%{value} %{label} tweets<extra></extra>",
        marker=dict(colors=[app_colors['text'], app_colors['plot2']])
    )

    layout = go.Layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=go.layout.Margin(
            t=40,
            b=0
        )
    )

    return go.Figure(
        data=data,
        layout=layout
    )

#%%
# wordcloud-----------------------------------------------------------------
image = Image.open(BytesIO(requests.get(LOGO).content))
processed_image = image.crop((70,70,330,330)).convert('L').resize((520, 360))
mask = np.array(ImageOps.invert(processed_image))

# stopwords.update()
def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(
        background_color=app_colors["plot4"],
        width=520, height=360, max_words=60,
        mask=mask,
        colormap='Blues')
    wc.fit_words(d)
    return wc.to_image()

@app.callback(
    Output('image-wc', 'src'), 
    [Input('term-search', 'n_clicks'),
    Input('wc-update', 'n_intervals'),
    Input('input-temp', 'children')]
)
def make_image(click, n, keyword):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        "SELECT tweet, keywords FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 2000",
        conn, params=('%' + keyword + '%', ))

    text = []
    for kw in df.keywords:
        seq = kw.split(',')
        seq = ' '.join(seq)
        text.append(seq)

    # word tokenize and stem
    text = [" ".join(tokenize(txt.lower())) for txt in text]
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.05)
    matrix = vectorizer.fit_transform(text).todense()
    # transform the matrix to a pandas df
    matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
    # sum over each document (axis=0)
    top_words = matrix.sum(axis=0).sort_values(ascending=False)

    dfm=pd.DataFrame(top_words.items(), columns=['word', 'freq'])
    rm_kw = [keyword.lower(), "not"]
    dfm=dfm[~ dfm['word'].str.contains('|'.join(rm_kw))]

    img = BytesIO()
    plot_wordcloud(data=dfm).save(img, format='PNG')

    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

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
            'border-bottom': '2px solid {}'.format(app_colors["plot4"]),
            'fontSize':14,
            'font-family':'Segoe UI'
        },
        style_cell={
            'backgroundColor':app_colors["text"],
            'color':app_colors["background"],
            'height': 'auto',
            'textAlign': 'center',
            'border': '1px solid {}'.format(app_colors["plot4"]),
            'fontSize':14,
            'font-family':'Segoe UI'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'tweet'},
            'width': '370px',
            'minWidth': '370px',
            'maxWidth': '370px',
            'whiteSpace': 'normal',
            'textAlign': 'left'
            }
        ],
        style_filter={
            'fontSize':14,
            'font-family':'sans-serif'
        },
        page_size=5,
        css=[
                {
                    'selector': '.previous-page, .next-page, .last-page, .first-page',
                    'rule': 'background-color: {};'.format(app_colors["plot4"])
                }
            ]
        )

@app.callback(
    Output('neg-live-table', 'children'),
    [Input('term-search', 'n_clicks'),
    Input('neg-table-update', 'n_intervals'),
    Input('input-temp', 'children')]
)
def update_negative_tweets(click, n, keyword):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        'SELECT tweet, sentiment FROM tweets WHERE tweet LIKE ? ORDER BY sentiment LIMIT 30',
        conn, params=('%' + keyword + '%', )
    )
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"RT|@\S+|\Whttps:\S+|\Whttp:\S+|\.\.\.",' ', x))
    df['sentiment'] = df['sentiment'].apply(lambda x: re.sub("(?<=....)(.*?)(?=e.+)", '', str(x)) if x<=1e-4 else round(x, 4)).astype(float)
    return generate_table(df)

@app.callback(
    Output('model-output', 'children'),
    [Input('model-submit', 'n_clicks')],
    [State('model-input', 'value')]
)
def update_output(click, value):
    if click > 0:
        test = preprocess(value)[1]
        sentiment = float(model.predict(test, batch_size=1024)[0][0])
        if sentiment <= 1e-4:
            return re.sub("(?<=....)(.*?)(?=e.+)", '', str(sentiment))
        else:
            return round(sentiment, 4)

#%%
#most postive tweets-------------------------------------------------------
@app.callback(
    Output('postive-live-table', 'children'),
    [Input('term-search', 'n_clicks'),
    Input('postive-table-update', 'n_intervals'),
    Input('input-temp', 'children')]
)
def update_postive_tweets(click, n, keyword):
    conn = sqlite3.connect('twitter.db')
    df = pd.read_sql(
        'SELECT tweet, sentiment FROM tweets WHERE tweet LIKE ? ORDER BY sentiment DESC LIMIT 30',
        conn, params=('%' + keyword + '%', )
    )
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"RT|@\S+|\Whttps:\S+|\Whttp:\S+|\.\.\.",' ', x))
    df['sentiment'] = df['sentiment'].apply(lambda x: round(x, 4)).astype(float)
    return generate_table(df)

@app.callback(
    Output('model-output', 'children'),
    [Input('model-submit', 'n_clicks')],
    [State('model-input', 'value')]
)
def update_output(click, value):
    if click > 0:
        test = preprocess(value)[1]
        sentiment = float(model.predict(test, batch_size=1024)[0][0])
        return round(sentiment, 4)

#%%
app.layout = html.Div([header, body, footer])

if __name__ == '__main__':
    app.run_server(debug=True)
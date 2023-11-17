import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import yfinance as yf
import plotly.graph_objs as go
import datetime
from datetime import timedelta
import os
from pandas.tseries.offsets import BDay
from getDailyData import get_daily
load_dotenv()

# Get the data for daily first
data_daily, df_final_daily, final_row_daily = get_daily()

engine = create_engine(
        f"mysql+mysqldb://{os.getenv('DATABASE_USERNAME')}:" \
        f"{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}/" \
        f"{os.getenv('DATABASE')}?ssl_ca=ca-certificates.crt&ssl_mode=VERIFY_IDENTITY"
    )

q = '''SELECT * FROM results where AsOf > '2022-06-01'
'''

df_all_results = pd.read_sql_query(q, con=engine.connect())
df_all_results['AsOf'] = df_all_results['AsOf'].dt.tz_localize('America/New_York')

# Get historical data
spx = yf.Ticker('^GSPC')
prices = spx.history(interval='30m', start=df_all_results.index.min(), ) 
df_all_results2 = df_all_results.merge(prices.reset_index()[['Datetime','Open','High','Low','Close']], left_on = 'AsOf', right_on = 'Datetime')
df_all_results2['Color'] = df_all_results2['Predicted'].apply(lambda x: 'green' if x >=0.6 else 'red' if x < 0.4 else 'yellow')
df_all_results2['PredDir'] = df_all_results2['Predicted'].apply(lambda x: 'Up' if x >=0.6 else 'Down' if x < 0.4 else 'Neutral')

date_select = datetime.datetime.today() - BDay(5)

if 'date_select' not in st.session_state:
    st.session_state.date_select = date_select

date_select = st.date_input(
    'Select data for chart',
    value=date_select,
    min_value=data_daily.index[0],
    max_value=data_daily.index[-1]
)

# Load your data
df1 = df_all_results2.set_index('AsOf')
df1 = df1.loc[df1.index > str(date_select)]

dts = df1.groupby(df1.index.date).head(1).reset_index()['AsOf']
daily_closes = data_daily.loc[df1.index.date, 'PrevClose'].drop_duplicates().reset_index()
daily_closes['FirstBar'] = dts
levels = data_daily.loc[df1.index.date, ['H1','H2','L1','L2','Open']].drop_duplicates().reset_index()
levels['FirstBar'] = dts

# Create a candlestick trace with custom colors based on the CandleColor column
candlestick_trace = go.Candlestick(
    x=df1.index,
    open=df1['Open'],
    high=df1['High'],
    low=df1['Low'],
    close=df1['Close'],
    increasing_fillcolor='#3399ff',
    decreasing_fillcolor='#ff5f5f',
    increasing_line_color='#3399ff',  # Color for decreasing candles
    decreasing_line_color='#ff5f5f',  # Color for decreasing candles
    name='30m'
)

df_up = df1.loc[df1['PredDir']=='Up']
df_down = df1.loc[df1['PredDir']=='Down']
df_neutral = df1.loc[df1['PredDir']=='Neutral']

scatter_up = go.Scatter(
    x=df_up.index,
    y=df_up['High'] * 1.001,
    mode='markers',
    marker=dict(size=8),
    marker_color=df_up['Color'],
    marker_symbol='triangle-up',
    name='Up'
)

scatter_down = go.Scatter(
    x=df_down.index,
    y=df_down['Low'] * 0.999,
    mode='markers',
    marker=dict(size=8),
    marker_color=df_down['Color'],
    marker_symbol='triangle-down',
    name='Down'
)

scatter_neut = go.Scatter(
    x=df_neutral.index,
    y=df_neutral[['Open','High','Low','Close']].mean(axis=1),
    mode='markers',
    marker=dict(size=7),
    marker_color=df_neutral['Color'],
    marker_symbol='diamond-open',
    name='Neutral'
)

# Create a layout
layout = go.Layout(
    title=dict(text='OHLC Chart with Predictions', xanchor='center', yanchor='top', y=0.9,x=0.5),

    xaxis=dict(title='Date'),
    yaxis=dict(title='Price'),
    template='plotly_dark',
    xaxis_rangeslider_visible=False,
    width=750,
    height=500
)

# Create a figure
fig = go.Figure(data=[candlestick_trace, scatter_up, scatter_neut, scatter_down], layout=layout)

fig.update_xaxes(
        rangebreaks=[
            # NOTE: Below values are bound (not single values), ie. hide x to y
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
            # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
        ]
    )

fig.update_layout(
    shapes = [dict(
        x0=d-timedelta(minutes=15), x1=d-timedelta(minutes=15), y0=0, y1=1, xref='x', yref='paper',
        line_width=0.5, opacity=0.5, line_dash='dot') for d in df1.loc[df1['ModelNum']==0].index],
    legend=dict(yanchor="top", y=1.05, xanchor="center", x=0.5, orientation='h'),
    margin=dict(l=20, r=20, t=80, b=20)
)


# Define the y-positions for your horizontal lines
pairs = [(start, level) for start, level in zip(daily_closes['FirstBar'], daily_closes['PrevClose'])]

# Add horizontal lines to the figure
for pair in pairs:
    start = pair[0] 
    end = start + BDay(1) - timedelta(minutes=15)
    level = pair[1]
    fig.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=level,
        y1=level,
        xref='x',
        yref='y',
        line=dict(
            width=0.5,
            dash="dot",
        ),
    )

for start in levels['FirstBar']:
    end = start + BDay(1)-timedelta(minutes=15)
    vals = levels.loc[levels['FirstBar']==start, ['H1','H2','L1','L2','Open']].values[0]
    H1 = vals[0]
    H2 = vals[1]
    L1 = vals[2]
    L2 = vals[3]
    Open = vals[4]
    # Plot H1
    fig.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=H1,
        y1=H1,
        xref='x',
        yref='y',
        line=dict(
            width=0.5,
            dash="solid",
            color="#ff5f5f"
            ),
        )
    # Plot H2
    fig.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=H2,
        y1=H2,
        xref='x',
        yref='y',
        line=dict(
            width=1,
            dash="solid",
            color="#ff5f5f"
            ),
        )
    # Plot L1
    fig.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=L1,
        y1=L1,
        xref='x',
        yref='y',
        line=dict(
            width=0.5,
            dash="solid",
            color="#3399ff"
            ),
        )
    # Plot L2
    fig.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=L2,
        y1=L2,
        xref='x',
        yref='y',
        line=dict(
            width=1,
            dash="solid",
            color="#3399ff"
            ),
        )
    # Plot Open
    fig.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=Open,
        y1=Open,
        xref='x',
        yref='y',
        line=dict(
            width=1,
            dash="solid",
            color="#cccccc"
            ),
        )

fig.for_each_xaxis(lambda x: x.update(showgrid=False))
fig.for_each_yaxis(lambda x: x.update(showgrid=False))

# Show the figure
st.plotly_chart(fig, use_container_width=True)

# Important levels
df_levels = pd.DataFrame(levels[['H2','H1','Open','L1','L2']].iloc[-1]).round(2)
df_levels.columns = ['Levels']
df_levels.astype(float).round(2)

# For historical reference
df_all_results['Symbol'] = df_all_results['Predicted'].apply(lambda x: '🟩' if x >=0.6 else '🟥' if x < 0.4 else '🟨')
today_df = df_all_results[['AsOf','Symbol','Predicted','CalibPredicted','Pvalue']].tail(13)[::-1]
today_df = today_df.set_index('AsOf', drop=True)
df_show = (today_df.style
           .format(formatter={
               'Predicted':'{:.1%}',
               'CalibPredicted':'{:.1%}',
               'Pvalue':'{:.2f}',
               })
)


st.dataframe(df_levels.T,use_container_width=True)
st.dataframe(df_show,use_container_width=True)
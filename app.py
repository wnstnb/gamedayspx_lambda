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
from streamlit_lightweight_charts import renderLightweightCharts
load_dotenv()

st.title('🎮 GamedaySPX Data Monitor')

# Get the data for daily first
data_daily, df_final_daily, final_row_daily = get_daily()

# Get historical data
spx = yf.Ticker('^GSPC')
prices = spx.history(interval='30m') 

date_select = st.date_input(
    'Select data for chart',
    value=datetime.datetime.today() - BDay(5),
    min_value=prices.index[0],
)

engine = create_engine(
        f"mysql+mysqldb://{os.getenv('DATABASE_USERNAME')}:" \
        f"{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}/" \
        f"{os.getenv('DATABASE')}?ssl_ca=ca-certificates.crt&ssl_mode=VERIFY_IDENTITY"
    )

q = f'''SELECT AsOf, Predicted, CalibPredicted, Pvalue, ModelNum FROM results where AsOf >= '{date_select}'
'''

df_all_results = pd.read_sql_query(q, con=engine.connect())
df_all_results['AsOf'] = df_all_results['AsOf'].dt.tz_localize('America/New_York')


df_all_results2 = df_all_results.merge(prices.reset_index()[['Datetime','Open','High','Low','Close']], left_on = 'AsOf', right_on = 'Datetime')
df_all_results2['Color'] = df_all_results2['Predicted'].apply(lambda x: 'green' if x >=0.6 else 'red' if x < 0.4 else 'yellow')
df_all_results2['PredDir'] = df_all_results2['Predicted'].apply(lambda x: 'Up' if x >=0.6 else 'Down' if x < 0.4 else 'Neutral')


# Load your data
df1 = df_all_results2.set_index('AsOf')
df1 = df1.loc[df1.index > str(date_select)]

dts = df1.groupby(df1.index.date).head(1).reset_index()['AsOf']
daily_closes = data_daily.loc[df1.index.date, 'PrevClose'].drop_duplicates().reset_index()
daily_closes['FirstBar'] = dts
levels = data_daily.loc[df1.index.date, ['H1','H2','L1','L2','Open']].drop_duplicates().reset_index()
levels['FirstBar'] = dts

# Plot
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

chartOptions = [{
    "width":800,
    "height":400,
    "rightPriceScale": {
        "scaleMargins": {
            "top": 0.2,
            "bottom": 0.25,
        },
        "borderVisible": False,
    },
    "layout": {
        "textColor": 'white',
        "background": {
            "type": 'solid',
            "color": 'black'
        },
    },
    "grid": {
        "vertLines": {
            "color": "rgba(197, 203, 206, 0)"
            },
        "horzLines": {
            "color": "rgba(197, 203, 206, 0)"
        }
    }
},
{
    "width":800,
    "height":125,
    "layout": {
        "textColor": 'white',
        "background": {
            "type": 'solid',
            "color": 'black'
        },
    },
    "grid": {
            "vertLines": {
                "color": "rgba(197, 203, 206, 0)"
                },
            "horzLines": {
                "color": "rgba(197, 203, 206, 0)"
            }
        },
},]

seriesCandlestickChart = [{
    
    "type": 'Candlestick',
    "data": [
        {"open": open, 
         "high": high, 
         "low": low, 
         "close": close, 
         "time": dt.timestamp()} for open, high, low, close, dt in zip(df1['Open'],df1['High'],df1['Low'],df1['Close'], df1.index)
    ],
    "options": {
        "upColor": '#3399ff',
        "downColor": '#ff5f5f',
        "borderVisible": False,
        "wickUpColor": '#3399ff',
        "wickDownColor": '#ff5f5f',
        "priceScaleVisible": True
    },
    "priceScale": {
        "scaleMargins": {
            "top": 0.7,
            "bottom": 0,
        }
    }
},
{
        "type": 'Line',
        "data": [{"value": value, "time":dt.timestamp()} for value, dt in zip(levels['H1'], levels['FirstBar'])],
        "options": {
            "color": 'blue',
            "lineWidth": 1
        }
    }]

seriesPredictions = [{
    "type": 'Histogram',
    "data": [
        { "value": pred, "time": dt.timestamp(), "color":color  } for pred, dt, color in zip(df1['CalibPredicted'], df1.index, df1['Color'])
    ],
    "options": { "color": '#26a69a' }
}]

renderLightweightCharts([
    {
        "chart": chartOptions[0],
        "series": seriesCandlestickChart
    },
    {
        "chart": chartOptions[1],
        "series": seriesPredictions
    },
], 'multipane')

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
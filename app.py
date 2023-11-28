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
# from streamlit_lightweight_charts import renderLightweightCharts
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

q = f'''SELECT 
            r.AsOf, 
            r.Predicted, 
            r.CalibPredicted, 
            r.Pvalue, 
            r.ModelNum,
            p.Predicted AS reg_pred,
            p.Upper,
            p.Lower
            FROM results r
            LEFT JOIN reg_results p ON r.AsOf = p.AsOf 
            where r.AsOf >= '{date_select}'
'''

df_all_results = pd.read_sql_query(q, con=engine.connect())
df_all_results['AsOf'] = df_all_results['AsOf'].dt.tz_localize('America/New_York')


df_all_results2 = df_all_results.merge(prices.reset_index()[['Datetime','Open','High','Low','Close']], left_on = 'AsOf', right_on = 'Datetime')
df_all_results2['Color'] = df_all_results2['Predicted'].apply(lambda x: 'green' if x >=0.6 else 'red' if x < 0.4 else 'yellow')
df_all_results2['PredDir'] = df_all_results2['Predicted'].apply(lambda x: 'Up' if x >=0.6 else 'Down' if x < 0.4 else 'Neutral')


# Load your data
df1 = df_all_results2.set_index('AsOf')
df1 = df1.loc[df1.index > str(date_select)]

import pytz
est = pytz.timezone('US/Eastern')
utc = pytz.utc

dts = df1.groupby(df1.index.date).head(1).reset_index()['AsOf']
daily_closes = data_daily.loc[df1.index.date, 'PrevClose'].drop_duplicates().reset_index()
daily_closes['FirstBar'] = dts
levels = data_daily.loc[df1.index.date, ['H1','H2','L1','L2','Open','PrevClose']].drop_duplicates().reset_index()
levels['FirstBar'] = dts
# levels['time'] = [dt.astimezone(est) for dt in levels['FirstBar']]
levels['time'] = levels['FirstBar'].copy()

# Plot

import streamlit as st
# from streamlit_lightweight_charts import renderLightweightCharts
from lightweight_charts.widgets import StreamlitChart

import numpy as np
import yfinance as yf
import pandas as pd


COLOR_BULL = '#ffffff' # #26a69a
COLOR_BEAR = '#787b86'  # #ef5350


# Some data wrangling to match required format
df = df1.copy()
df['time'] = [dt for dt in df.index]
# df['time'] = [dt.timestamp() for dt in df.index]
# df = df[['time','Open','High','Low','Close','CalibPredicted','Color','Upper','Lower','reg_pred']].bfill()
# df.columns = ['time','open','high','low','close','volume','color','Upper','Lower','reg_pred']                  # rename columns
df = df[['time','Open','High','Low','Close','CalibPredicted','Upper','Lower','reg_pred']].bfill()
df.columns = ['time','open','high','low','close','volume','Upper','Lower','reg_pred']                  # rename columns

df = df.merge(levels, how = 'left', on = 'time')
# df['time'] = [dt.timestamp() for dt in df['time']]
df[['H1','H2','L1','L2','Open','PrevClose']] = df[['H1','H2','L1','L2','Open','PrevClose']].ffill()
df['UpperP'] = (df['Upper'] + 1) * df['PrevClose']
df['RegPred'] = (df['reg_pred'] + 1) * df['PrevClose']
df['LowerP'] = (df['Lower'] + 1) * df['PrevClose']

chart = StreamlitChart(
    height=350,
    width=700,
    scale_candles_only=True,
    )
chart.layout(background_color='rgba(214, 237, 255, 0)', 
             text_color='#787b86', font_size=10,
                 font_family='Arial')

chart.candle_style(up_color='#ffffff', down_color='#787b86',
                       border_up_color='#ffffff', border_down_color='#787b86',
                       wick_up_color='#ffffff', wick_down_color='#787b86')

chart.volume_config(up_color='rgba(255,255,255,0.4)', down_color='rgba(255,255,255,0.4)',scale_margin_top=.9)

# chart.legend(visible=True, font_size=14)
chart.grid(vert_enabled=False, horz_enabled=False)
chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted',
                    horz_color='#FFFFFF', horz_style='dotted')

df = df[['time','open','high','low','close','volume','H1','H2','L1','L2','Open','PrevClose']]
# Lines to make
_H1 = chart.create_line('H1', color='#ffb8b8', price_line=False)
_H1.set(df[['H1']])
_H2 = chart.create_line('H2', color='#ffb8b8', price_line=False)
_H2.set(df[['H2']])
_L1 = chart.create_line('L1', color='#96cbff', price_line=False)
_L1.set(df[['L1']])
_L2 = chart.create_line('L2', color='#96cbff', price_line=False)
_L2.set(df[['L2']])
_Open = chart.create_line('Open', color='#ffffff', style='dotted', price_line=False)
_Open.set(df[['Open']])
_PrevClose = chart.create_line('PrevClose', color='#C724B1', price_line=False)
_PrevClose.set(df[['PrevClose']])

chart.legend(visible=True, ohlc = True, color = '#ffffff', font_size=10, font_family='Calibri')
chart.set(df)
chart.set_visible_range(df['time'].min(), df['time'].max())
chart.load()

# export to JSON format
# candles = json.loads(json.dumps([
#         {"open": open, 
#          "high": high, 
#          "low": low, 
#          "close": close, 
#          "time": dt} for open, high, low, close, dt in zip(df['open'],df['high'],df['low'],df['close'], df['time'])
#     ], indent=2))
# # volume = json.loads(df.rename(columns={"volume": "value",}).to_json(orient = "records"))
# volume = json.loads(json.dumps([
#         { "value": pred, "time": dt, "color":color  } for pred, dt, color in zip(df['volume'], df['time'], df['color'])
#     ], indent=2))

# h1 = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['H1'], df['time'])
#     ], indent=2))

# h2 = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['H2'], df['time'])
#     ], indent=2))

# opens = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['Open'], df['time'])
#     ], indent=2)) 

# l1 = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['L1'], df['time'])
#     ], indent=2)) 
    
# l2 = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['L2'], df['time'])
#     ], indent=2)) 

# uppers = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['UpperP'], df['time'])
#     ], indent=2)) 

# lowers = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['LowerP'], df['time'])
#     ], indent=2)) 

# regPreds = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['RegPred'], df['time'])
#     ], indent=2)) 

# prevCloses = json.loads(json.dumps([
#         { "value": h1, "time": dt  } for h1, dt in zip(df['PrevClose'], df['time'])
#     ], indent=2)) 

# chartMultipaneOptions = [
#     {
#         # "width": 600,
#         "height": 300,
#         "layout": {
#             "background": {
#                 "type": "solid",
#                 "color": 'transparent'
#             },
#             "textColor": "white"
#         },
#         "grid": {
#             "vertLines": {
#                 "color": "rgba(197, 203, 206, 0)"
#                 },
#             "horzLines": {
#                 "color": "rgba(197, 203, 206, 0)"
#             }
#         },
#         "crosshair": {
#             "mode": 0
#         },
#         "priceScale": {
#             "borderColor": "rgba(197, 203, 206, 0.8)",
#             "autoScale": True,
#             "scaleMargins": {
#                 "top": 0,
#                 "bottom": 0
#             },
#             "alignLabels": True
#         },
        
#         "timeScale": {
#             "borderColor": "rgba(197, 203, 206, 0.8)",
#             "barSpacing": 5,
#             "timeVisible": True
#         }
#     },
#     {
#         # "width": 600,
#         "height": 75,
#         "layout": {
#             "background": {
#                 "type": 'solid',
#                 "color": 'transparent'
#             },
#             "textColor": 'white',
#         },
#         "grid": {
#             "vertLines": {
#                 "color": 'rgba(42, 46, 57, 0)',
#             },
#             "horzLines": {
#                 "color": 'rgba(42, 46, 57, 0.6)',
#             }
#         },
#         "timeScale": {
#             "visible": True,
#         }
#     },
#     {
#         "width": 600,
#         "height": 200,
#         "layout": {
#             "background": {
#                 "type": "solid",
#                 "color": 'white'
#             },
#             "textColor": "black"
#         },
#         "timeScale": {
#             "visible": False,
#         },
#         "watermark": {
#             "visible": True,
#             "fontSize": 18,
#             "horzAlign": 'left',
#             "vertAlign": 'center',
#             "color": 'rgba(171, 71, 188, 0.7)',
#             "text": 'MACD',
#         }
#     }
# ]

# seriesCandlestickChart = [
#     {
#         "type": 'Candlestick',
#         "data": candles,
#         "options": {
#             "upColor": COLOR_BULL,
#             "downColor": COLOR_BEAR,
#             "borderVisible": False,
#             "wickUpColor": COLOR_BULL,
#             "wickDownColor": COLOR_BEAR
#         }
#     },
#     {
#         "type": 'Line',
#         "data": h1,
#         "options": {
#             "color": '#ffb8b8',
#             "lineWidth": 1,
#             "lineType": 1,
#             "lineStyle": 4,
#             "priceLineVisible": False
#         }
#     },
#     {
#         "type": 'Line',
#         "data": h2,
#         "options": {
#             "color": '#ffb8b8',
#             "lineWidth": 1,
#             "lineType": 1,
#             "lineStyle": 4,
#             "priceLineVisible": False
#         }
#     },
#     {
#         "type": 'Line',
#         "data": opens,
#         "options": {
#             "color": '#ffffff',
#             "lineWidth": 1,
#             "lineType": 1,
#             "lineStyle": 0,
#             "priceLineVisible": False
#         }
#     },
#     {
#         "type": 'Line',
#         "data": l1,
#         "options": {
#             "color": '#96cbff',
#             "lineWidth": 1,
#             "lineType": 1,
#             "lineStyle": 4,
#             "priceLineVisible": False
#         }
#     },
#     {
#         "type": 'Line',
#         "data": l2,
#         "options": {
#             "color": '#96cbff',
#             "lineWidth": 1,
#             "lineType": 1,
#             "lineStyle": 4,
#             "priceLineVisible": False,
#         }
#     },
#     {
#         "type": 'Line',
#         "data": prevCloses,
#         "options": {
#             "color": '#C724B1',
#             "lineWidth": 1,
#             "lineType": 1,
#             "lineStyle": 0,
#             "priceLineVisible": False
#         }
#     },
#     {
#         "type": 'Histogram',
#         "data": volume,
#         "options": {
#             "priceFormat": {
#                 "type": 'volume',
#             },
#             "priceScaleId": "", # set as an overlay setting,
#             "priceLineVisible": False
#         },
#         "priceScale": {
#             "scaleMargins": {
#                 "top": 0.9,
#                 "bottom": 0,
#             },
#         }
#     }
# ]

# renderLightweightCharts([
#     {
#         "chart": chartMultipaneOptions[0],
#         "series": seriesCandlestickChart
#     }
# ], 'priceAndVolume')

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
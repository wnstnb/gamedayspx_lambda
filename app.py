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
levels['time'] = levels['FirstBar'].apply(lambda x: x.timestamp())

# Plot

import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import json
import numpy as np
import yfinance as yf
import pandas as pd


COLOR_BULL = '#ffffff' # #26a69a
COLOR_BEAR = '#787b86'  # #ef5350


# Some data wrangling to match required format
df = df1.copy()
df['time'] = [dt.timestamp() for dt in df.index]
df = df[['time','Open','High','Low','Close','CalibPredicted','Color','Upper','Lower','reg_pred']].bfill()
df.columns = ['time','open','high','low','close','volume','color','Upper','Lower','reg_pred']                  # rename columns
df = df.merge(levels, how = 'left', on = 'time')
# df['time'] = [dt.timestamp() for dt in df['time']]
df[['H1','H2','L1','L2','Open','PrevClose']] = df[['H1','H2','L1','L2','Open','PrevClose']].ffill()
df['UpperP'] = (df['Upper'] + 1) * df['PrevClose']
df['RegPred'] = (df['reg_pred'] + 1) * df['PrevClose']
df['LowerP'] = (df['Lower'] + 1) * df['PrevClose']
# export to JSON format
# candles = json.loads(df.to_json(orient = "records"))
candles = json.loads(json.dumps([
        {"open": open, 
         "high": high, 
         "low": low, 
         "close": close, 
         "time": dt} for open, high, low, close, dt in zip(df['open'],df['high'],df['low'],df['close'], df['time'])
    ], indent=2))
# volume = json.loads(df.rename(columns={"volume": "value",}).to_json(orient = "records"))
volume = json.loads(json.dumps([
        { "value": pred, "time": dt, "color":color  } for pred, dt, color in zip(df['volume'], df['time'], df['color'])
    ], indent=2))

h1 = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['H1'], df['time'])
    ], indent=2))

h2 = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['H2'], df['time'])
    ], indent=2))

opens = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['Open'], df['time'])
    ], indent=2)) 

l1 = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['L1'], df['time'])
    ], indent=2)) 
    
l2 = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['L2'], df['time'])
    ], indent=2)) 

uppers = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['UpperP'], df['time'])
    ], indent=2)) 

lowers = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['LowerP'], df['time'])
    ], indent=2)) 

regPreds = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['RegPred'], df['time'])
    ], indent=2)) 

prevCloses = json.loads(json.dumps([
        { "value": h1, "time": dt  } for h1, dt in zip(df['PrevClose'], df['time'])
    ], indent=2)) 

chartMultipaneOptions = [
    {
        # "width": 600,
        "height": 300,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'transparent'
            },
            "textColor": "white"
        },
        "grid": {
            "vertLines": {
                "color": "rgba(197, 203, 206, 0)"
                },
            "horzLines": {
                "color": "rgba(197, 203, 206, 0)"
            }
        },
        "crosshair": {
            "mode": 0
        },
        "priceScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "autoScale": True,
            "scaleMargins": {
                "above": 0,
                "below": 0
            },
            "alignLabels": True
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "barSpacing": 15,
            "timeVisible": True
        }
    },
    {
        # "width": 600,
        "height": 75,
        "layout": {
            "background": {
                "type": 'solid',
                "color": 'transparent'
            },
            "textColor": 'black',
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(42, 46, 57, 0)',
            },
            "horzLines": {
                "color": 'rgba(42, 46, 57, 0.6)',
            }
        },
        "timeScale": {
            "visible": True,
        }
    },
    {
        "width": 600,
        "height": 200,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'white'
            },
            "textColor": "black"
        },
        "timeScale": {
            "visible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 18,
            "horzAlign": 'left',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.7)',
            "text": 'MACD',
        }
    }
]

seriesCandlestickChart = [
    {
        "type": 'Candlestick',
        "data": candles,
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR
        }
    },
    {
        "type": 'Line',
        "data": h1,
        "options": {
            "color": '#ffb8b8',
            "lineWidth": 1,
            "lineType": 1,
            "lineStyle": 4,
            "priceLineVisible": False
        }
    },
    {
        "type": 'Line',
        "data": h2,
        "options": {
            "color": '#ffb8b8',
            "lineWidth": 1,
            "lineType": 1,
            "lineStyle": 4,
            "priceLineVisible": False
        }
    },
    {
        "type": 'Line',
        "data": opens,
        "options": {
            "color": '#ffffff',
            "lineWidth": 1,
            "lineType": 1,
            "lineStyle": 0,
            "priceLineVisible": False
        }
    },
    {
        "type": 'Line',
        "data": l1,
        "options": {
            "color": '#96cbff',
            "lineWidth": 1,
            "lineType": 1,
            "lineStyle": 4,
            "priceLineVisible": False
        }
    },
    {
        "type": 'Line',
        "data": l2,
        "options": {
            "color": '#96cbff',
            "lineWidth": 1,
            "lineType": 1,
            "lineStyle": 4,
            "priceLineVisible": False,
        }
    },
    {
        "type": 'Line',
        "data": prevCloses,
        "options": {
            "color": '#cccccc',
            "lineWidth": 1,
            "lineType": 1,
            "lineStyle": 4,
            "priceLineVisible": False
        }
    },
    {
        "type": 'Histogram',
        "data": volume,
        "options": {
            "priceFormat": {
                "type": 'volume',
            },
            "priceScaleId": "" # set as an overlay setting,
        },
        "priceScale": {
            "scaleMargins": {
                "top": 0.85,
                "bottom": 0,
            },
        }
    }
    # {
    #     "type": 'Line',
    #     "data": lowers,
    #     "options": {
    #         "color": '#ffffff',
    #         "lineWidth": 1,
    #         "lineType": 0,
    #         "priceLineVisible": False
    #     }
    # }
]

seriesVolumeChart = [
    {
        "type": 'Histogram',
        "data": volume,
        "options": {
            "priceFormat": {
                "type": 'volume',
            },
            "priceScaleId": "" # set as an overlay setting,
        },
        "priceScale": {
            "scaleMargins": {
                "top": 0.1,
                "bottom": 0.1,
            },
            "alignLabels": True
        }
    }
]

renderLightweightCharts([
    {
        "chart": chartMultipaneOptions[0],
        "series": seriesCandlestickChart
    }
], 'priceAndVolume')
# import streamlit as st
# from streamlit_lightweight_charts import renderLightweightCharts

# chartOptions = [{
#     "width":800,
#     "height":400,
#     "rightPriceScale": {
#         "scaleMargins": {
#             "top": 0.2,
#             "bottom": 0.25,
#         },
#         "borderVisible": False,
#     },
#     "overlayPriceScales": {
#         "scaleMargins": {
#             "top": 0.7,
#             "bottom": 0,
#         }
#     },
#     "layout": {
#         "textColor": 'white',
#         "background": {
#             "type": 'solid',
#             "color": 'black'
#         },
#     },
#     "grid": {
#         "vertLines": {
#             "color": "rgba(197, 203, 206, 0)"
#             },
#         "horzLines": {
#             "color": "rgba(197, 203, 206, 0)"
#         }
#     }
# },
# {
#     "width":800,
#     "height":125,
#     "layout": {
#         "textColor": 'white',
#         "background": {
#             "type": 'solid',
#             "color": 'black'
#         },
#     },
#     "grid": {
#             "vertLines": {
#                 "color": "rgba(197, 203, 206, 0)"
#                 },
#             "horzLines": {
#                 "color": "rgba(197, 203, 206, 0)"
#             }
#         },
# },]

# seriesCandlestickChart = [{
    
#     "type": 'Candlestick',
#     "data": [
#         {"open": open, 
#          "high": high, 
#          "low": low, 
#          "close": close, 
#          "time": dt.timestamp()} for open, high, low, close, dt in zip(df1['Open'],df1['High'],df1['Low'],df1['Close'], df1.index)
#     ],
#     "options": {
#         "upColor": '#3399ff',
#         "downColor": '#ff5f5f',
#         "borderVisible": False,
#         "wickUpColor": '#3399ff',
#         "wickDownColor": '#ff5f5f',
#         "priceScaleVisible": True
#     },
#     "priceScale": {
#         "scaleMargins": {
#             "top": 0.7,
#             "bottom": 0,
#         }
#     }
# },
# {
#         "type": 'Line',
#         "data": [{"value": value, "time":dt.timestamp()} for value, dt in zip(levels['H1'], levels['FirstBar'])],
#         "options": {
#             "color": 'blue',
#             "lineWidth": 1
#         }
#     }]

# seriesPredictions = [{
#     "type": 'Histogram',
#     "data": [
#         { "value": pred, "time": dt.timestamp(), "color":color  } for pred, dt, color in zip(df1['CalibPredicted'], df1.index, df1['Color'])
#     ],
#     "options": { "color": '#26a69a' }
# }]

# renderLightweightCharts([
#     {
#         "chart": chartOptions[0],
#         "series": seriesCandlestickChart
#     },
#     {
#         "chart": chartOptions[1],
#         "series": seriesPredictions
#     },
# ], 'multipane')

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
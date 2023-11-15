import pandas as pd
import yfinance as yf
import datetime
from getDailyData import data_start_date
from dbConn import engine

def get_intra(periods_30m = 1):
    '''
    Method to get historical 30 minute data and append live data to it, if exists. 
    '''

    query = f'''SELECT
        spx30.Datetime AS Datetime,
        spx30.Open AS Open30,
        spx30.High AS High30,
        spx30.Low AS Low30,
        spx30.Close AS Close30,
        vix30.Open AS Open_VIX30,
        vix30.High AS High_VIX30,
        vix30.Low AS Low_VIX30,
        vix30.Close AS Close_VIX30,
        vvix30.Open AS Open_VVIX30,
        vvix30.High AS High_VVIX30,
        vvix30.Low AS Low_VVIX30,
        vvix30.Close AS Close_VVIX30
    FROM 
        SPX_full_30min AS spx30
    LEFT JOIN 
        VIX_full_30min AS vix30 ON spx30.Datetime = vix30.Datetime AND vix30.Datetime > '{data_start_date}'
    LEFT JOIN 
        VVIX_full_30min AS vvix30 ON spx30.Datetime = vvix30.Datetime AND vvix30.Datetime > '{data_start_date}'
    WHERE 
        spx30.Datetime > '{data_start_date}'

    '''

    df_30m = pd.read_sql_query(sql=query, con=engine.connect())
    df_30m['Datetime'] = df_30m['Datetime'].dt.tz_localize('America/New_York')
    df_30m = df_30m.set_index('Datetime',drop=True)

    # Get incremental date
    last_date = df_30m.index.date[-1]
    last_date = last_date + datetime.timedelta(days=1)

    # Get incremental data for each index
    spx1 = yf.Ticker('^GSPC')
    vix1 = yf.Ticker('^VIX')
    vvix1 = yf.Ticker('^VVIX')    
    yfp = spx1.history(start=last_date, interval='30m')
    yf_vix = vix1.history(start=last_date, interval='30m')
    yf_vvix = vvix1.history(start=last_date, interval='30m')

    if len(yfp) > 0:
        # Convert indexes to EST if not already
        for _df in [yfp, yf_vix, yf_vvix]:
            if (_df.index.tz.zone != 'America/New_York') or (type(_df.index) != pd.DatetimeIndex):
                _df['Datetime'] = pd.to_datetime(_df.index)
                _df['Datetime'] = _df['Datetime'].dt.tz_convert('America/New_York')
                _df.set_index('Datetime', inplace=True)
        # Concat them 
        df_inc = pd.concat([
            yfp[['Open','High','Low','Close']], 
            yf_vix[['Open','High','Low','Close']], 
            yf_vvix[['Open','High','Low','Close']]
            ], axis=1)
        df_inc.columns = df_30m.columns
        df_inc = df_inc.loc[
            (df_inc.index.time >= datetime.time(9,30)) & (df_inc.index.time < datetime.time(16,00))
        ]
        df_30m = pd.concat([df_30m, df_inc])
    else:
        df_30m = df_30m.copy()

    df_30m = df_30m.loc[
                (df_30m.index.time >= datetime.time(9,30)) & (df_30m.index.time < datetime.time(16,00))
            ]
    df_30m['dt'] = df_30m.index.date
    df_30m = df_30m.groupby('dt').head(periods_30m)
    df_30m = df_30m.set_index('dt',drop=True)
    df_30m.index.name = 'Datetime'

    df_30m['SPX30IntraPerf'] = (df_30m['Close30'] / df_30m['Close30'].shift(1)) - 1
    df_30m['VIX30IntraPerf'] = (df_30m['Close_VIX30'] / df_30m['Close_VIX30'].shift(1)) - 1
    df_30m['VVIX30IntraPerf'] = (df_30m['Close_VVIX30'] / df_30m['Close_VVIX30'].shift(1)) - 1

    opens_intra = df_30m.groupby('Datetime')[[c for c in df_30m.columns if 'Open' in c]].head(1)
    highs_intra = df_30m.groupby('Datetime')[[c for c in df_30m.columns if 'High' in c]].max()
    lows_intra = df_30m.groupby('Datetime')[[c for c in df_30m.columns if 'Low' in c]].min()
    closes_intra = df_30m.groupby('Datetime')[[c for c in df_30m.columns if 'Close' in c]].tail(1)
    spx_intra = df_30m.groupby('Datetime')['SPX30IntraPerf'].tail(1)
    vix_intra = df_30m.groupby('Datetime')['VIX30IntraPerf'].tail(1)
    vvix_intra = df_30m.groupby('Datetime')['VVIX30IntraPerf'].tail(1)

    df_intra = pd.concat([opens_intra, highs_intra, lows_intra, closes_intra, spx_intra, vix_intra, vvix_intra], axis=1)
    return df_intra
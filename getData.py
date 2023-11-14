import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from typing import List
from tqdm import tqdm
import os
import datetime

model_cols = [
    'BigNewsDay',
    'Quarter',
    'Perf5Day',
    'Perf5Day_n1',
    'DaysGreen',
    'DaysRed',
    'CurrentHigh30toClose',
    'CurrentLow30toClose',
    'CurrentClose30toClose',
    'CurrentRange30',
    'GapFill30',
    'CurrentGap',
    'RangePct',
    'RangePct_n1',
    'RangePct_n2',
    'OHLC4_VIX',
    'OHLC4_VIX_n1',
    'OHLC4_VIX_n2',
    'OHLC4_Current_Trend',
    'OHLC4_Trend',
    'CurrentVIXTrend',
    'SPX30IntraPerf',
    'VIX30IntraPerf',
    'VVIX30IntraPerf',
    # 'OpenL1',
    # 'OpenL2',
    # 'OpenH1',
    # 'OpenH2',
    'L1TouchPct',
    'L2TouchPct',
    'H1TouchPct',
    'H2TouchPct',
    'L1BreakPct',
    'L2BreakPct',
    'H1BreakPct',
    'H2BreakPct',
    'GreenProbas',
    'H1BreakTouchPct',
    'H2BreakTouchPct',
    'L1BreakTouchPct',
    'L2BreakTouchPct',
    'H1BreakH2TouchPct',
    'L1BreakL2TouchPct',
    'H1TouchGreenPct',    
    'L1TouchRedPct'    
    # 'GapFillGreenProba'
]

def get_data(periods_30m = 1):
    # f = open('settings.json')
    # j = json.load(f)
    # API_KEY_FRED = j["API_KEY_FRED"]

    API_KEY_FRED = os.getenv('API_KEY_FRED')
    
    def parse_release_dates(release_id: str) -> List[str]:
        release_dates_url = f'https://api.stlouisfed.org/fred/release/dates?release_id={release_id}&realtime_start=2015-01-01&include_release_dates_with_no_data=true&api_key={API_KEY_FRED}'
        r = requests.get(release_dates_url)
        text = r.text
        soup = BeautifulSoup(text, 'xml')
        dates = []
        for release_date_tag in soup.find_all('release_date', {'release_id': release_id}):
            dates.append(release_date_tag.text)
        return dates

    econ_dfs = {}

    econ_tickers = [
        'WALCL',
        'NFCI',
        'WRESBAL'
    ]

    for et in tqdm(econ_tickers, desc='getting econ tickers'):
        df = pdr.get_data_fred(et)
        df.index = df.index.rename('ds')
        econ_dfs[et] = df

    release_ids = [
        "10", # "Consumer Price Index"
        "46", # "Producer Price Index"
        "50", # "Employment Situation"
        "53", # "Gross Domestic Product"
        "103", # "Discount Rate Meeting Minutes"
        "180", # "Unemployment Insurance Weekly Claims Report"
        "194", # "ADP National Employment Report"
        "323" # "Trimmed Mean PCE Inflation Rate"
    ]

    release_names = [
        "CPI",
        "PPI",
        "NFP",
        "GDP",
        "FOMC",
        "UNEMP",
        "ADP",
        "PCE"
    ]

    releases = {}

    for rid, n in tqdm(zip(release_ids, release_names), total = len(release_ids), desc='Getting release dates'):
        releases[rid] = {}
        releases[rid]['dates'] = parse_release_dates(rid)
        releases[rid]['name'] = n 

    # Create a DF that has all dates with the name of the col as 1
    # Once merged on the main dataframe, days with econ events will be 1 or None. Fill NA with 0
    # This column serves as the true/false indicator of whether there was economic data released that day.
    for rid in tqdm(release_ids, desc='Making indicators'):
        releases[rid]['df'] = pd.DataFrame(
            index=releases[rid]['dates'],
            data={
            releases[rid]['name']: 1
            })
        releases[rid]['df'].index = pd.DatetimeIndex(releases[rid]['df'].index)

    vix = yf.Ticker('^VIX')
    vvix = yf.Ticker('^VVIX')
    spx = yf.Ticker('^GSPC')

    # Pull in data
    data_files = {"spx": "SPX_full_30min.txt", "vix": "VIX_full_30min.txt", "vvix":'VVIX_full_30min.txt'}
    data = load_dataset("boomsss/spx_intra", data_files=data_files)
    dfs = []
    for ticker in data.keys():
        rows = [d['text'] for d in data[ticker]]
        rows = [x.split(',') for x in rows]

        fr = pd.DataFrame(columns=[
            'Datetime','Open','High','Low','Close'
        ], data = rows)

        fr['Datetime'] = pd.to_datetime(fr['Datetime'])
        fr['Datetime'] = fr['Datetime'].dt.tz_localize('America/New_York')
        fr = fr.set_index('Datetime')
        fr['Open'] = pd.to_numeric(fr['Open'])
        fr['High'] = pd.to_numeric(fr['High'])
        fr['Low'] = pd.to_numeric(fr['Low'])
        fr['Close'] = pd.to_numeric(fr['Close'])
        dfs.append(fr)

    df_30m = pd.concat(dfs, axis=1)

    df_30m.columns = [
        'Open30',
        'High30',
        'Low30',
        'Close30',
        'Open_VIX30',
        'High_VIX30',
        'Low_VIX30',
        'Close_VIX30',
        'Open_VVIX30',
        'High_VVIX30',
        'Low_VVIX30',
        'Close_VVIX30'
    ]

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
            if _df.index.tz.zone != 'America/New_York':
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


    prices_vix = vix.history(start=data_start_date, interval='1d')
    prices_vvix = vvix.history(start=data_start_date, interval='1d')
    prices_spx = spx.history(start=data_start_date, interval='1d')

    prices_spx['index'] = [str(x).split()[0] for x in prices_spx.index]
    prices_spx['index'] = pd.to_datetime(prices_spx['index']).dt.date
    prices_spx.index = prices_spx['index']
    prices_spx = prices_spx.drop(columns='index')
    prices_spx.index = pd.DatetimeIndex(prices_spx.index)

    prices_vix['index'] = [str(x).split()[0] for x in prices_vix.index]
    prices_vix['index'] = pd.to_datetime(prices_vix['index']).dt.date
    prices_vix.index = prices_vix['index']
    prices_vix = prices_vix.drop(columns='index')
    prices_vix.index = pd.DatetimeIndex(prices_vix.index)

    prices_vvix['index'] = [str(x).split()[0] for x in prices_vvix.index]
    prices_vvix['index'] = pd.to_datetime(prices_vvix['index']).dt.date
    prices_vvix.index = prices_vvix['index']
    prices_vvix = prices_vvix.drop(columns='index')
    prices_vvix.index = pd.DatetimeIndex(prices_vvix.index)

    data = prices_spx.merge(df_intra, left_index=True, right_index=True)
    data = data.merge(prices_vix[['Open','High','Low','Close']], left_index=True, right_index=True, suffixes=['','_VIX'])
    data = data.merge(prices_vvix[['Open','High','Low','Close']], left_index=True, right_index=True, suffixes=['','_VVIX'])

    # Features
    data['PrevClose'] = data['Close'].shift(1)
    data['Perf5Day'] = data['Close'] > data['Close'].shift(5)
    data['Perf5Day_n1'] = data['Perf5Day'].shift(1)
    data['Perf5Day_n1'] = data['Perf5Day_n1'].astype(bool)
    data['GreenDay'] = (data['Close'] > data['PrevClose']) * 1
    data['RedDay'] = (data['Close'] <= data['PrevClose']) * 1

    data['VIX5Day'] = data['Close_VIX'] > data['Close_VIX'].shift(5)
    data['VIX5Day_n1'] = data['VIX5Day'].astype(bool)

    data['VVIX5Day'] = data['Close_VVIX'] > data['Close_VVIX'].shift(5)
    data['VVIX5Day_n1'] = data['VVIX5Day'].astype(bool)

    data['Range'] = data[['Open','High']].max(axis=1) - data[['Low','Open']].min(axis=1) # Current day range in points
    data['RangePct'] = data['Range'] / data['Close']
    data['VIXLevel'] = pd.qcut(data['Close_VIX'], 4)
    data['OHLC4_VIX'] = data[['Open_VIX','High_VIX','Low_VIX','Close_VIX']].mean(axis=1)
    data['OHLC4'] = data[['Open','High','Low','Close']].mean(axis=1)
    data['OHLC4_Trend'] = data['OHLC4'] > data['OHLC4'].shift(1)
    data['OHLC4_Trend'] = data['OHLC4_Trend'].astype(bool)
    data['OHLC4_Trend_n1'] = data['OHLC4_Trend'].shift(1)
    data['OHLC4_Trend_n1'] = data['OHLC4_Trend_n1'].astype(float)
    data['OHLC4_Trend_n2'] = data['OHLC4_Trend'].shift(1)
    data['OHLC4_Trend_n2'] = data['OHLC4_Trend_n2'].astype(float)
    data['RangePct_n1'] = data['RangePct'].shift(1)
    data['RangePct_n2'] = data['RangePct'].shift(2)
    data['OHLC4_VIX_n1'] = data['OHLC4_VIX'].shift(1)
    data['OHLC4_VIX_n2'] = data['OHLC4_VIX'].shift(2)
    data['CurrentGap'] = (data['Open'] - data['PrevClose']) / data['PrevClose']
    data['CurrentGapHist'] = data['CurrentGap'].copy()
    data['CurrentGap'] = data['CurrentGap'].shift(-1)
    data['DayOfWeek'] = pd.to_datetime(data.index)
    data['DayOfWeek'] = data['DayOfWeek'].dt.day

    # Intraday features
    data['CurrentOpen30'] = data['Open30'].shift(-1)
    data['CurrentHigh30'] = data['High30'].shift(-1)
    data['CurrentLow30'] = data['Low30'].shift(-1)
    data['CurrentClose30'] = data['Close30'].shift(-1)
    data['CurrentOHLC430'] = data[['CurrentOpen30','CurrentHigh30','CurrentLow30','CurrentClose30']].max(axis=1)
    data['OHLC4_Current_Trend'] = data['CurrentOHLC430'] > data['OHLC4']
    data['OHLC4_Current_Trend'] = data['OHLC4_Current_Trend'].astype(bool)
    data['HistClose30toPrevClose'] = (data['Close30'] / data['PrevClose']) - 1

    data['CurrentCloseVIX30'] = data['Close_VIX30'].shift(-1)
    data['CurrentOpenVIX30'] = data['Open_VIX30'].shift(-1)

    data['CurrentVIXTrend'] = data['CurrentCloseVIX30'] > data['Close_VIX']

    # Open to High
    data['CurrentHigh30toClose'] = (data['CurrentHigh30'] / data['Close']) - 1
    data['CurrentLow30toClose'] = (data['CurrentLow30'] / data['Close']) - 1
    data['CurrentClose30toClose'] = (data['CurrentClose30'] / data['Close']) - 1
    data['CurrentRange30'] = (data['CurrentHigh30'] - data['CurrentLow30']) / data['Close']
    data['GapFill30'] = [low <= prev_close if gap > 0 else high >= prev_close for high, low, prev_close, gap in zip(data['CurrentHigh30'], data['CurrentLow30'], data['Close'], data['CurrentGap'])]
    
    # Target -- the next day's low
    data['Target'] = (data['OHLC4'] / data['PrevClose']) - 1
    data['Target'] = data['Target'].shift(-1)
    # data['Target'] = data['RangePct'].shift(-1)

    # Target for clf -- whether tomorrow will close above or below today's close
    data['Target_clf'] = data['Close'] > data['PrevClose']
    data['Target_clf'] = data['Target_clf'].shift(-1)
    data['DayOfWeek'] = pd.to_datetime(data.index)
    data['Quarter'] = data['DayOfWeek'].dt.quarter
    data['DayOfWeek'] = data['DayOfWeek'].dt.weekday

    # Calculate up
    data['up'] = 100 * (data['High'].shift(1) - data['Open'].shift(1)) / data['Close'].shift(1)

    # Calculate upSD
    data['upSD'] = data['up'].rolling(30).std(ddof=0)

    # Calculate aveUp
    data['aveUp'] = data['up'].rolling(30).mean()
    data['H1'] = data['Open'] + (data['aveUp'] / 100) * data['Open']
    data['H2'] = data['Open'] + ((data['aveUp'] + data['upSD']) / 100) * data['Open']
    data['down'] = 100 * (data['Open'].shift(1) - data['Low'].shift(1)) / data['Close'].shift(1)
    data['downSD'] = data['down'].rolling(30).std(ddof=0)
    data['aveDown'] = data['down'].rolling(30).mean()
    data['L1'] = data['Open'] - (data['aveDown'] / 100) * data['Open']
    data['L2'] = data['Open'] - ((data['aveDown'] + data['downSD']) / 100) * data['Open']

    data = data.assign(
        L1Touch = lambda x: x['Low'] < x['L1'],
        L2Touch = lambda x: x['Low'] < x['L2'],
        H1Touch = lambda x: x['High'] > x['H1'],
        H2Touch = lambda x: x['High'] > x['H2'],
        L1Break = lambda x: x['Close'] < x['L1'],
        L1TouchRed = lambda x: (x['Low'] < x['L2']) & (x['Close'] < x['PrevClose']),
        L2TouchL1Break = lambda x: (x['Low'] < x['L2']) & (x['Close'] < x['L1']),
        L2Break = lambda x: x['Close'] < x['L2'],
        H1Break = lambda x: x['Close'] > x['H1'],
        H1TouchGreen = lambda x: (x['High'] > x['H1']) & (x['Close'] > x['PrevClose']),
        H2TouchH1Break = lambda x: (x['High'] > x['H2']) & (x['Close'] > x['H1']),
        H2Break = lambda x: x['Close'] > x['H2'],
        OpenL1 = lambda x: np.where(x['Open'] < x['L1'], 1, 0),
        OpenL2 = lambda x: np.where(x['Open'] < x['L2'], 1, 0),
        OpenH1 = lambda x: np.where(x['Open'] > x['H1'], 1, 0),
        OpenH2 = lambda x: np.where(x['Open'] > x['H2'], 1, 0),
        CloseL1 = lambda x: np.where(x['Close30'] < x['L1'], 1, 0),
        CloseL2 = lambda x: np.where(x['Close30'] < x['L2'], 1, 0),
        CloseH1 = lambda x: np.where(x['Close30'] > x['H1'], 1, 0),
        CloseH2 = lambda x: np.where(x['Close30'] > x['H2'], 1, 0)
    )

    data['OpenL1'] = data['OpenL1'].shift(-1)
    data['OpenL2'] = data['OpenL2'].shift(-1)
    data['OpenH1'] = data['OpenH1'].shift(-1)
    data['OpenH2'] = data['OpenH2'].shift(-1)
    data['CloseL1'] = data['CloseL1'].shift(-1)
    data['CloseL2'] = data['CloseL2'].shift(-1)
    data['CloseH1'] = data['CloseH1'].shift(-1)
    data['CloseH2'] = data['CloseH2'].shift(-1)

    level_cols = [
        'L1Touch',
        'L2Touch',
        'H1Touch',
        'H2Touch',
        'L1Break',
        'L2Break',
        'H1Break',
        'H2Break'
    ]

    for col in level_cols:
        data[col+'Pct'] = data[col].rolling(100).mean()
        # data[col+'Pct'] = data[col+'Pct'].shift(-1)

    data['H1BreakTouchPct'] = data['H1Break'].rolling(100).sum() / data['H1Touch'].rolling(100).sum()
    data['H2BreakTouchPct'] = data['H2Break'].rolling(100).sum() / data['H2Touch'].rolling(100).sum()
    data['L1BreakTouchPct'] = data['L1Break'].rolling(100).sum() / data['L1Touch'].rolling(100).sum()
    data['L2BreakTouchPct'] = data['L2Break'].rolling(100).sum() / data['L2Touch'].rolling(100).sum()
    data['L1TouchRedPct'] = data['L1TouchRed'].rolling(100).sum() / data['L1Touch'].rolling(100).sum()
    data['H1TouchGreenPct'] = data['H1TouchGreen'].rolling(100).sum() / data['H1Touch'].rolling(100).sum()

    data['H1BreakH2TouchPct'] = data['H2TouchH1Break'].rolling(100).sum() / data['H2Touch'].rolling(100).sum()
    data['L1BreakL2TouchPct'] = data['L2TouchL1Break'].rolling(100).sum() / data['L2Touch'].rolling(100).sum()

    def get_quintiles(df, col_name, q):
        return df.groupby(pd.qcut(df[col_name], q))['GreenDay'].mean()

    probas = []
    # Given the current price level
    for i, pct in enumerate(data['CurrentClose30toClose']):
        try:
            # Split
            df_q = get_quintiles(data.iloc[:i], 'HistClose30toPrevClose', 10)
            for q in df_q.index:
                if q.left <= pct <= q.right:
                    p = df_q[q]
        except:
            p = None

        probas.append(p)

    # gapfills = []
    # for i, pct in enumerate(data['CurrentGap']):
    #     try:
    #         df_q = get_quintiles(data.iloc[:i], 'CurrentGapHist', 5)
    #         for q in df_q.index:
    #             if q.left <= pct <= q.right:
    #                 p = df_q[q]
    #     except:
    #         p = None

    #     gapfills.append(p)

    data['GreenProbas'] = probas
    # data['GapFillGreenProba'] = gapfills

    for rid in tqdm(release_ids, desc='Merging econ data'):
        # Get the name of the release
        n = releases[rid]['name']
        # Merge the corresponding DF of the release
        data = data.merge(releases[rid]['df'], how = 'left', left_index=True, right_index=True)
        # Create a column that shifts the value in the merged column up by 1
        data[f'{n}_shift'] = data[n].shift(-1)
        # Fill the rest with zeroes
        data[n] = data[n].fillna(0)
        data[f'{n}_shift'] = data[f'{n}_shift'].fillna(0)
        
    data['BigNewsDay'] = data[[x for x in data.columns if '_shift' in x]].max(axis=1)

    def cumul_sum(col):
        nums = []
        s = 0
        for x in col:
            if x == 1:
                s += 1
            elif x == 0:
                s = 0
            nums.append(s)
        return nums

    consec_green = cumul_sum(data['GreenDay'].values)
    consec_red = cumul_sum(data['RedDay'].values)

    data['DaysGreen'] = consec_green
    data['DaysRed'] = consec_red

    final_row = data.index[-2]

    exp_row = data.index[-1]

    df_final = data.loc[:final_row, model_cols + ['Target', 'Target_clf']]
    df_final = df_final.dropna(subset=['Target','Target_clf'])
    # df_final = df_final.dropna(subset=['Target','Target_clf','Perf5Day_n1'])
    return data, df_final, final_row
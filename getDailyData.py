import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from pandas.tseries.offsets import BDay
from dbConn import engine

data_start_date = '2018-07-01'

def get_daily(mode='daily', periods_30m=None):
    '''
    Method to get daily data and create daily features. Optionally append intra data if specified.
    `mode`: 'daily' or 'intra'.
    `periods_30m`: How many 30m periods to bring in. Only specify if mode == 'intra'.
    '''
    
    vix = yf.Ticker('^VIX')
    vvix = yf.Ticker('^VVIX')
    spx = yf.Ticker('^GSPC')

    query = f'''SELECT
        spx.Datetime AS Datetime,
        spx.Open AS Open,
        spx.High AS High,
        spx.Low AS Low,
        spx.Close AS Close,
        vix.Open AS Open_VIX,
        vix.High AS High_VIX,
        vix.Low AS Low_VIX,
        vix.Close AS Close_VIX,
        vvix.Open AS Open_VVIX,
        vvix.High AS High_VVIX,
        vvix.Low AS Low_VVIX,
        vvix.Close AS Close_VVIX
    FROM 
        SPX_full_1day AS spx
    LEFT JOIN 
        VIX_full_1day AS vix ON spx.Datetime = vix.Datetime AND vix.Datetime > '{data_start_date}'
    LEFT JOIN 
        VVIX_full_1day AS vvix ON spx.Datetime = vvix.Datetime AND vvix.Datetime > '{data_start_date}'
    WHERE 
        spx.Datetime > '{data_start_date}'

    '''
    data = pd.read_sql_query(sql=query, con=engine.connect())
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime',drop=True)

    # Get incremental date
    last_date = data.index.date[-1]
    last_date = last_date + BDay(1) 

    prices_vix = vix.history(start=last_date, interval='1d')
    prices_vvix = vvix.history(start=last_date, interval='1d')
    prices_spx = spx.history(start=last_date, interval='1d')

    if len(prices_spx) > 0:

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

        data1 = prices_spx.merge(prices_vix[['Open','High','Low','Close']], left_index=True, right_index=True, suffixes=['','_VIX'])
        data1 = data1.merge(prices_vvix[['Open','High','Low','Close']], left_index=True, right_index=True, suffixes=['','_VVIX'])
        data = pd.concat([data, data1])

    else:
        data = data.copy()

    if mode == 'intra':
        from getIntraData import get_intra
        df_intra = get_intra(periods_30m)
        data = data.merge(df_intra, left_index=True, right_index=True)
    else:
        data = data.copy()

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

    data['VIXOpen'] = data['Open_VIX'] > data['Close_VIX'].shift(1)
    data['VVIXOpen'] = data['Open_VVIX'] > data['Close_VVIX'].shift(1)
    data['VIXOpen'] = data['VIXOpen'].astype(bool)
    data['VVIXOpen'] = data['VVIXOpen'].astype(bool)

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

    # Target -- the next day's low
    data['Target'] = (data['OHLC4'] / data['PrevClose']) - 1
    data['Target'] = data['Target'].shift(-1)
    # data['Target'] = data['RangePct'].shift(-1)

    # Target for clf -- whether tomorrow will close above or below today's close
    data['Target_clf'] = data['Close'] > data['PrevClose']
    data['ClosePct'] = (data['Close'] / data['PrevClose']) - 1
    data['ClosePct'] =  data['ClosePct'].shift(-1)
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
        OpenH2 = lambda x: np.where(x['Open'] > x['H2'], 1, 0)
    )

    data['OpenL1'] = data['OpenL1'].shift(-1)
    data['OpenL2'] = data['OpenL2'].shift(-1)
    data['OpenH1'] = data['OpenH1'].shift(-1)
    data['OpenH2'] = data['OpenH2'].shift(-1)


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

    if mode=='intra':
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
        data['CloseL1'] = np.where(data['Close30'] < data['L1'], 1, 0)
        data['CloseL2'] = np.where(data['Close30'] < data['L2'], 1, 0)
        data['CloseH1'] = np.where(data['Close30'] > data['H1'], 1, 0)
        data['CloseH2'] = np.where(data['Close30'] > data['H2'], 1, 0)
        data['CloseL1'] = data['CloseL1'].shift(-1)
        data['CloseL2'] = data['CloseL2'].shift(-1)
        data['CloseH1'] = data['CloseH1'].shift(-1)
        data['CloseH2'] = data['CloseH2'].shift(-1)

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

        data['GreenProbas'] = probas

    df_releases = pd.read_sql_query('select * from releases', con=engine)
    df_releases = df_releases.set_index('Datetime')
    data = data.merge(df_releases, how = 'left', left_index=True, right_index=True)

    for n in tqdm(df_releases.columns, desc='Merging econ data'):
        # Get the name of the release
        # n = releases[rid]['name']
        # Merge the corresponding DF of the release
        # data = data.merge(releases[rid]['df'], how = 'left', left_index=True, right_index=True)
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

    if mode=='daily':
        from dailyCols import model_cols

    elif mode=='intra':
        from intraCols import model_cols
        from regrCols import model_cols as regr_cols

    df_final = data.loc[:final_row, model_cols + ['Target', 'Target_clf', 'ClosePct']]
    df_final = df_final.dropna(subset=['Target','Target_clf'])
    # df_final = df_final.dropna(subset=['Target','Target_clf','Perf5Day_n1'])
    return data, df_final, final_row
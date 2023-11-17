#!/usr/bin/env python3

# Function should get the data and run the whole model, return a single prediction based on the time
from getDailyData import get_daily
from model_intra_v3 import walk_forward_validation
from model_day_v2 import walk_forward_validation_seq as walk_forward_validation_daily
import pandas as pd
import json
from dbConn import connection, engine, insert_dataframe_to_sql
import numpy as np
from datetime import time, timedelta
import datetime
from pandas.tseries.offsets import BDay
import holidays
from dotenv import load_dotenv
load_dotenv()

def is_trading_day_and_time():
    # Get the current local time (assuming the machine is set to EST)
    now = datetime.datetime.now()

    # Check if today is a weekend (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return False

    # Check if today is a US public holiday
    us_holidays = holidays.US()
    if now.date() in us_holidays:
        return False

    # Define trading hours (9:30 AM to 4:00 PM)
    start_time = datetime.time(9, 30, 0)
    end_time = datetime.time(16, 0, 0)

    # Check if current time is within trading hours
    if now.time() >= start_time and now.time() <= end_time:
        return True
    else:
        return False
    
def is_refresh_time():
    # Get the current local time (assuming the machine is set to EST)
    now = datetime.datetime.now()

    # Check if today is a weekend (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return False

    # Check if today is a US public holiday
    us_holidays = holidays.US()
    if now.date() in us_holidays:
        return False

    # Define trading hours (9:30 AM to 4:00 PM)
    refresh_time = datetime.time(16, 15, 0)

    # Check if current time is within trading hours
    if now.time() >= refresh_time:
        return True
    else:
        return False


def lambda_handler(periods_30m):
    if periods_30m > 0:
        data, df_final, final_row = get_daily(mode='intra', periods_30m=periods_30m)
        res, _ = walk_forward_validation(df_final.drop(columns=['Target']).dropna(), 'Target_clf', 1, mode='single')

    elif periods_30m == 0:
        data, df_final, final_row = get_daily()
        res, _, _ = walk_forward_validation_daily(df_final.dropna(), 'Target_clf', 'Target', 200, 1)
    
    # Get results, run calibration and pvalue    
    df_results = pd.read_sql_query(f'select * from results where ModelNum = {str(periods_30m)}', con = engine)

    # Calibrate Probabilities
    def get_quantiles(df, col_name, q):
        return df.groupby(pd.cut(df[col_name], q))['IsTrue'].mean()

    pct = res['Predicted'].iloc[-1]

    df_q = get_quantiles(df_results, 'Predicted', 10)
    for q in df_q.index:
        if q.left <= pct <= q.right:
            p = df_q[q]

    calib_scores = np.abs(df_results['Predicted'].iloc[:-1] - 0.5)
    score = abs(pct - 0.5)
    pv = np.mean(calib_scores >= score)
    asof = datetime.datetime.combine(data.index[-1], time(9,30)) + (periods_30m * timedelta(minutes=30)) 

    blob = {
        'Datetime': str(res.index[-1]),
        'IsTrue':df_final['Target_clf'].iloc[-1],
        'Predicted': pct,
        'CalibPredicted': p,
        'Pvalue':pv,
        'ModelNum':periods_30m,
        'AsOf':str(asof)
    }

    # Write to DB
    df_write = pd.DataFrame.from_dict({k:[v] for k, v in blob.items()})
    cursor = connection.cursor()
    insert_dataframe_to_sql('results', df_write, cursor)
    # cursor.close()
    # connection.close()

    return blob

if __name__ == '__main__':
    # Code that, based on the time of the day, return which data/model to run
    game_time = is_trading_day_and_time()
    refresh_time = is_refresh_time()
    if game_time:
        now = datetime.datetime.now()
        # Change this for debugging -- should be EST
        morning_start = datetime.datetime.combine(now.date(), time(9, 30))
        delta = now - morning_start
        print(delta)
        intervals = max(0,min((delta.total_seconds() / 60 / 30) // 1, 12))
        print(f'running for {str(intervals)}')
        j = lambda_handler(intervals)
    elif refresh_time:
        times_list = np.arange(0,13)
        for i in times_list: 
            j = lambda_handler(i)
            # print(j)
    else:
        print("It's either a weekend, a holiday, or outside RTH. Do not run the script.")
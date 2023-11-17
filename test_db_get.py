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

def test_db():
    # Get results, run calibration and pvalue    
    df_results = pd.read_sql_query(f'select * from results where ModelNum = 12', con = engine)
    res = df_results.iloc[-1]
    return res

if __name__ == '__main__':
    p = test_db()
    with open("/home/ec2-user/logfile.log", "a") as file:
        file.write(f"Cron job ran at {datetime.datetime.now()}\n{str(p)}")
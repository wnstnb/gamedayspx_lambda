import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

import datetime
from datetime import time, timedelta
from tqdm import tqdm

def prep_data(df):
morning_start = datetime.datetime.combine(now.date(), time(6, 30))
delta = now - morning_start
print(delta)
# candle = 1 #max(0,min((delta.total_seconds() / 60 / 30) // 1, 12))
# candles = np.arange(1,13)
candles = np.arange(1,2)
for candle in tqdm(candles):
    print(f'running for {str(candle)}')
    data, df_final, final_row = get_daily(mode='intra', periods_30m=candle)

    df_new = data[['Open','High','Low','Close','Close30','Close_VIX30','Close_VIX','Close_VVIX30','Close_VVIX']].copy()
    df_new['PrevClose'] = df_new['Close'].shift(1)
    df_new['CurrentGap'] = (df_new['Open'] / df_new['PrevClose']) - 1
    df_new['ClosePctIntra'] = (df_new['Close30'] / df_new['Close'].shift(1)) - 1
    df_new['ClosePctOpenIntra'] = (df_new['Close30'] / df_new['Open']) - 1
    df_new['ClosePctVIXIntra'] = (df_new['Close_VIX30'] / df_new['Close_VIX'].shift(1)) - 1
    df_new['ClosePctVVIXIntra'] = (df_new['Close_VVIX30'] / df_new['Close_VVIX'].shift(1)) - 1
    df_new['EMA8'] = df_new['Close'].ewm(8).mean()
    df_new['EMA8'] = df_new['EMA8'].shift(1)
    df_new['EMA8Intra'] = df_new['Close30'] > df_new['EMA8']

    # Target will be the day's close
    df_new['ClosePct'] = (df_new['Close'] / df_new['Close'].shift(1)) - 1

    # Column to determine what percentile the current intra performance looks like
    intra_rank = []
    for i, pct in tqdm(enumerate(df_new['ClosePctIntra'])):
        try:
            historical = df_new['ClosePctIntra'].iloc[:i]
            current = df_new['ClosePctIntra'].iloc[i]
            perc = len(historical[historical > current]) / len(historical)
        except:
            perc = None
        intra_rank.append(perc)

    df_new['IntraPercentile'] = intra_rank

    # Column to determine what percentile the daily performance looks like
    daily_rank = []
    for i, pct in tqdm(enumerate(df_new['ClosePct'])):
        try:
            historical = df_new['ClosePct'].iloc[:i]
            current = df_new['ClosePct'].iloc[i]
            perc = len(historical[historical > current]) / len(historical)
        except:
            perc = None
        daily_rank.append(perc)

    df_new['ClosePctPercentile'] = daily_rank

    # Let's do n-5 to start just for closes
    lags = np.arange(1,6)

    for lag in lags:
        df_new[f'ClosePct_n{str(lag)}'] = df_new['ClosePct'].shift(lag)
        # df_new[f'ClosePctPercentile_n{str(lag)}'] = df_new['ClosePctPercentile'].shift(lag)


    df_feats = df_new[[c for c in df_new.columns if 'ClosePct' in c or 'Intra' in c or 'Gap' in c]]

    df_final = df_feats.dropna()

    X = df_final[['ClosePctIntra']]  # Feature dataset
    y = df_final['ClosePct']    # Target dataset

    # model = LGBMRegressor(random_state=42, n_estimators=10, verbose=-1)
    # model = LinearRegression()
    # Define the column transformer for handling numeric and categorical features
    

    # Fit the pipeline on the training data
    # pipeline.fit(X_train, y_train)

    tscv = TimeSeriesSplit(n_splits=len(df_final)-1, max_train_size=None, test_size=1)

    mae_scores = []
    overall_results = []

    for train_index, test_index in tscv.split(X):
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        # Select features
        categorical_features = X_train.select_dtypes(include='object').columns
        numeric_features = X_train.drop(columns=[c for c in X_train.columns if 'Percentile' in c]).select_dtypes(include='number').columns

        # Transformers
        numeric_transformer = RobustScaler()  # Example: StandardScaler for numeric features
        categorical_transformer = OneHotEncoder()  # Example: OneHotEncoder for categorical features

        # Define the pipeline steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),  # numeric_features is a list of numeric feature column names
                ('categorical', categorical_transformer, categorical_features)  # categorical_features is a list of categorical feature column names
            ])

        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        # mae_scores.append(mean_absolute_error(y_test, y_pred))
        result_df = pd.DataFrame({'IsTrue': y_test, 'Predicted': y_pred}, index=y_test.index)
        overall_results.append(result_df)

    df_results = pd.concat(overall_results)

    uppers = []
    lowers = []
    alpha = 0.05
    for i, pct in tqdm(enumerate(df_results['Predicted']), desc='Calibrating Probas',total=len(df_results)):
        try:
            
            df_q = df_results.iloc[:i]
            pred = df_results['Predicted'].iloc[-1]
            errors = df_q['IsTrue'] - df_q['Predicted']
            positive_errors = errors[errors >= 0]
            negative_errors = errors[errors < 0]

            # Calculate bounds
            upper_bound = pred + np.quantile(positive_errors, 1 - alpha)
            lower_bound = pred + np.quantile(negative_errors, alpha)
            
        except:
            upper_bound = None
            lower_bound = None

        uppers.append(upper_bound)
        lowers.append(lower_bound)

    df_results['Upper'] = uppers
    df_results['Lower'] = lowers

    df_results = df_results.merge(data[['PrevClose']],left_index=True, right_index=True)
    df_results['Pred'] = df_results['PrevClose'] * (1 + df_results['Predicted'])
    df_results['Actual'] = df_results['PrevClose'] * (1 + df_results['IsTrue'])
    df_results['Up'] = df_results['PrevClose'] * (1 + df_results['Upper'])
    df_results['Down'] = df_results['PrevClose'] * (1 + df_results['Lower'])

    results[f'{str(int(candle))}'] = df_results

    # Average metrics across folds
    average_mae = mean_absolute_error(df_results['IsTrue'], df_results['Predicted'])
    # sorted_features = sorted([(feat, coef) for feat, coef in zip(model.feature_name_, model.feature_importances_)], key=lambda x: abs(x[1]), reverse=True)
    sorted_features = sorted([(feat, coef) for feat, coef in zip(pipeline.feature_names_in_, pipeline.named_steps.model.coef_)], key=lambda x: abs(x[1]), reverse=True)

    coefs[f'{str(int(candle))}'] = pd.DataFrame(sorted_features, columns=['Feature','Coefficient'])

    df_consolidated.loc[int(candle), 'MAE'] = average_mae
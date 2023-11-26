import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from regrCols import model_cols

def walk_forward_validation(df, target_column, num_periods, mode='full'):
    
    df = df[model_cols + [target_column]]
    df[target_column] = df[target_column].astype(float)

    tscv = TimeSeriesSplit(n_splits=len(df)-1, max_train_size=None, test_size=num_periods)  # num_splits is the number of splits you want

    if mode == 'full':
        overall_results = []
        # Iterate over the rows in the DataFrame, one step at a time
        # Split the time series data using TimeSeriesSplit
        for train_index, test_index in tqdm(tscv.split(df), total=tscv.n_splits):
            # Extract the training and testing data for the current split
            X_train = df.drop(target_column, axis=1).iloc[train_index]
            y_train = df[target_column].iloc[train_index]
            X_test = df.drop(target_column, axis=1).iloc[test_index]
            y_test = df[target_column].iloc[test_index]
        
            y_train = y_train.astype(float)
            model = LinearRegression()
            model.fit(X_train, y_train)
            # Make a prediction on the test data
            predictions = model.predict(X_test)
                
            # Create a DataFrame to store the true and predicted values
            result_df = pd.DataFrame({'IsTrue': y_test, 'Predicted': predictions}, index=y_test.index)
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

    elif mode == 'single':
        X_train = df.drop(target_column, axis=1).iloc[:-1]
        y_train = df[target_column].iloc[:-1]
        X_test = df.drop(target_column, axis=1).iloc[-1]
        y_test = df[target_column].iloc[-1]
        y_train = y_train.astype(float)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test.values.reshape(1, -1))
        df_results = pd.DataFrame({'IsTrue': y_test, 'Predicted': predictions}, index=[df.index[-1]])

    return df_results, model
        
def calc_upper_lower(pred, df_hist, alpha=0.05):
    errors = df_hist['IsTrue'] - df_hist['Predicted']
    positive_errors = errors[errors >= 0]
    negative_errors = errors[errors < 0]

    # Calculate bounds
    upper_bound = pred + np.quantile(positive_errors, 1 - alpha)
    lower_bound = pred + np.quantile(negative_errors, alpha)

    return upper_bound, lower_bound


def seq_predict_proba(df, trained_clf_model):
    clf_pred_proba = trained_clf_model.predict_proba(df[model_cols])[:,-1]
    return clf_pred_proba
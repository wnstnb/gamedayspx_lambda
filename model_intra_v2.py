import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from intraCols import model_cols

def walk_forward_validation(df, target_column, num_periods, mode='full'):
    
    df = df[model_cols + [target_column]]
    df[target_column] = df[target_column].astype(bool)

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
        
            y_train = y_train.astype(bool)
            model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
            model.fit(X_train, y_train)
            # Make a prediction on the test data
            predictions = model.predict_proba(X_test)[:,-1]
                
            # Create a DataFrame to store the true and predicted values
            result_df = pd.DataFrame({'IsTrue': y_test, 'Predicted': predictions}, index=y_test.index)
            overall_results.append(result_df)

        df_results = pd.concat(overall_results)
        
        # Calibrate Probabilities
        def get_quantiles(df, col_name, q):
            return df.groupby(pd.cut(df[col_name], q))['IsTrue'].mean()

        greenprobas = []
        pvals = []
        for i, pct in tqdm(enumerate(df_results['Predicted']), desc='Calibrating Probas',total=len(df_results)):
            try:
                df_q = get_quantiles(df_results.iloc[:i], 'Predicted', 10)
                for q in df_q.index:
                    if q.left <= pct <= q.right:
                        p = df_q[q]

                calib_scores = np.abs(df_results['Predicted'].iloc[:i] - 0.5)
                score = abs(df_results['Predicted'].iloc[i] - 0.5)
                pv = np.mean(calib_scores >= score)
            except:
                p = None
                pv = None

            greenprobas.append(p)
            pvals.append(pv)
        
        df_results['CalibPredicted'] = greenprobas
        df_results['Pvalue'] = pvals

    elif mode == 'single':
        X_train = df.drop(target_column, axis=1).iloc[:-1]
        y_train = df[target_column].iloc[:-1]
        X_test = df.drop(target_column, axis=1).iloc[-1]
        y_test = df[target_column].iloc[-1]
        y_train = y_train.astype(bool)
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test.values.reshape(1, -1))[:,-1]
        df_results = pd.DataFrame({'IsTrue': y_test, 'Predicted': predictions}, index=[df.index[-1]])

    return df_results, model
        

def seq_predict_proba(df, trained_clf_model):
    clf_pred_proba = trained_clf_model.predict_proba(df[model_cols])[:,-1]
    return clf_pred_proba
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
import lightgbm as lgb
from dailyCols import model_cols

def walk_forward_validation(df, target_column, num_training_rows, num_periods):
    
    # Create an XGBRegressor model
    # model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state = 42)
    model = linear_model.LinearRegression()

    overall_results = []
    # Iterate over the rows in the DataFrame, one step at a time
    for i in tqdm(range(num_training_rows, df.shape[0] - num_periods + 1),desc='LR Model'):
        # Split the data into training and test sets
        X_train = df.drop(target_column, axis=1).iloc[:i]
        y_train = df[target_column].iloc[:i]
        X_test = df.drop(target_column, axis=1).iloc[i:i+num_periods]
        y_test = df[target_column].iloc[i:i+num_periods]
        
        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        # Make a prediction on the test data
        predictions = model.predict(X_test)
        
        # Create a DataFrame to store the true and predicted values
        result_df = pd.DataFrame({'True': y_test, 'Predicted': predictions}, index=y_test.index)
        
        overall_results.append(result_df)

    df_results = pd.concat(overall_results)
    # model.save_model('model_lr.bin')
    # Return the true and predicted values, and fitted model
    return df_results, model

def walk_forward_validation_seq(df, target_column_clf, target_column_regr, num_training_rows, num_periods):

    # Create run the regression model to get its target
    res, model1 = walk_forward_validation(df.drop(columns=[target_column_clf]).dropna(), target_column_regr, num_training_rows, num_periods)
    # joblib.dump(model1, 'model1.bin')

    # Merge the result df back on the df for feeding into the classifier
    for_merge = res[['Predicted']]
    for_merge.columns = ['RegrModelOut']
    for_merge['RegrModelOut'] = for_merge['RegrModelOut'] > 0
    df = df.merge(for_merge, left_index=True, right_index=True)
    df = df.drop(columns=[target_column_regr])
    df = df[model_cols + ['RegrModelOut', target_column_clf]]
    
    df[target_column_clf] = df[target_column_clf].astype(bool)
    df['RegrModelOut'] = df['RegrModelOut'].astype(bool)

    # Create an XGBRegressor model
    # model2 = xgb.XGBClassifier(n_estimators=10, random_state = 42)
    model2 = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
    # model = linear_model.LogisticRegression(max_iter=1500)
    
    overall_results = []
    # Iterate over the rows in the DataFrame, one step at a time
    for i in tqdm(range(num_training_rows, df.shape[0] - num_periods + 1),'CLF Model'):
        # Split the data into training and test sets
        X_train = df.drop(target_column_clf, axis=1).iloc[:i]
        y_train = df[target_column_clf].iloc[:i]
        X_test = df.drop(target_column_clf, axis=1).iloc[i:i+num_periods]
        y_test = df[target_column_clf].iloc[i:i+num_periods]
        
        # Fit the model to the training data
        model2.fit(X_train, y_train)
        
        # Make a prediction on the test data
        predictions = model2.predict_proba(X_test)[:,-1]
        
        # Create a DataFrame to store the true and predicted values
        result_df = pd.DataFrame({'True': y_test, 'Predicted': predictions}, index=y_test.index)
        
        overall_results.append(result_df)

    df_results = pd.concat(overall_results)

    # Calibrate Probabilities
    def get_quantiles(df, col_name, q):
        return df.groupby(pd.cut(df[col_name], q))['True'].mean()

    greenprobas = []
    meanprobas = []
    for i, pct in tqdm(enumerate(df_results['Predicted']), desc='Calibrating Probas'):
        try:
            df_q = get_quantiles(df_results.iloc[:i], 'Predicted', 7)
            for q in df_q.index:
                if q.left <= pct <= q.right:
                    p = df_q[q]
                    c = (q.left + q.right) / 2
        except:
            p = None
            c = None

        greenprobas.append(p)
        meanprobas.append(c)

    df_results['CalibPredicted'] = greenprobas
    
    return df_results, model1, model2

def seq_predict_proba(df, trained_reg_model, trained_clf_model):
    regr_pred = trained_reg_model.predict(df)
    regr_pred = regr_pred > 0
    new_df = df.copy()
    new_df['RegrModelOut'] = regr_pred
    clf_pred_proba = trained_clf_model.predict_proba(new_df[model_cols + ['RegrModelOut']])[:,-1]
    return clf_pred_proba
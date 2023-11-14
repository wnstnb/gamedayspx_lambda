import pandas as pd
import lightgbm as lgb
from intraCols import model_cols

def walk_forward_validation(df, target_column, num_periods, mode='full'):
    
    df = df[model_cols + [target_column]]
    df[target_column] = df[target_column].astype(bool)

    X_train = df.drop(target_column, axis=1).iloc[:-1]
    y_train = df[target_column].iloc[:-1]
    X_test = df.drop(target_column, axis=1).iloc[-1]
    y_test = df[target_column].iloc[-1]
    y_train = y_train.astype(bool)
    model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test.values.reshape(1, -1))[:,-1]
    result_df = pd.DataFrame({'True': y_test, 'Predicted': predictions}, index=[df.index[-1]])

    return result_df, model
    
def seq_predict_proba(df, trained_clf_model):
    clf_pred_proba = trained_clf_model.predict_proba(df[model_cols])[:,-1]
    return clf_pred_proba
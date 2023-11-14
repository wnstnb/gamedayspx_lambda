# Function should get the data and run the whole model, return a single prediction based on the time
from getDailyData import get_daily
from model_intra_v3 import walk_forward_validation
import json

def lambda_handler(periods_30m):
    data, df_final, final_row = get_daily(mode='intra', periods_30m=periods_30m)
    res, _ = walk_forward_validation(df_final.drop(columns=['Target']).dropna(), 'Target_clf', 1, mode='single')
    return json.loads(json.dumps({
        'date': str(res.index[-1]),
        'prediction': res['Predicted'].iloc[-1],
        'time':periods_30m
    }))

if __name__ == '__main__':
    j = lambda_handler(1)
    print(j)
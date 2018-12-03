import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.cross_validation import KFold
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMClassifier

import gc
import os



y_train=np.load('../output/candidate/lgb_m4_y_train.npy')

x_train=np.empty(shape=[714167,1])
x_test=np.empty(shape=[617242,1])

filepath='/mnt/output/candidate/'
for file in os.listdir(filepath):
    if file.endswith("oof_train.npy"):
        print (file)
        oof_train =np.load(filepath+file).reshape(-1,1)
        x_train=np.concatenate((x_train,oof_train),axis=1)
        del oof_train
        gc.collect()


for file in os.listdir(filepath):
    if file.endswith("oof_test.npy"):
        oof_test =np.load(filepath+file).reshape(-1,1)
        x_test=np.concatenate((x_test,oof_test),axis=1)
        del oof_test
        gc.collect()

#np.save('x_train', x_train)
#np.save('x_test', x_test)
print('train length',x_train.shape)
print('test length',x_test.shape)

dtrain = xgb.DMatrix(x_train, label=np.log1p(y_train))
dtest = xgb.DMatrix(x_test)

xgb_params = {
                'objective': 'reg:linear',
                'metric': 'rmse',
                'booster': 'gbtree',
                'learning_rate': 0.02,
                'max_depth': 22,
                'min_child_weight': 57,
                'gamma' : 1.45,
                'alpha': 0.0,
                'lambda': 0.0,
                'subsample': 0.67,
                'colsample_bytree': 0.054,
                'colsample_bylevel': 0.50,
                'n_jobs': -1,
                'random_state': 456,
                'sead':0
}

print("xgb cv..")
res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=5, seed=17, stratified=False,
             early_stopping_rounds=200, verbose_eval=10, show_stdv=True)
best_nrounds = res.shape[0] - 1

print("meta xgb train..")
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
fi = gbdt.predict(dtest)
fi = np.array(fi)
#np.save('fi', fi)

subm=pd.read_csv('../output/submission/lgb_m4.csv')
subm['PredictedLogRevenue'] = fi
subm.to_csv('../output/submission/stack1_diff_data.csv', index=False)
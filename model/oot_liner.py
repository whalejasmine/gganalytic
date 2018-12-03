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
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import gc
import os



y_train=np.load('../output/candidate/lgb_m4_y_train.npy')

x_train=np.empty(shape=[714167,2])
x_test=np.empty(shape=[617242,2])


for file in os.listdir('../output/candidate/'):
    if file.endswith("*_oof_train.npy"):
        oof_train =np.load(file)
        #oof_train.reshape(-1,1)
        x_train=np.concatenate((x_train,oof_train),axis=1)
        del oof_train
        gc.collect()


for file in os.listdir('../output/candidate/'):
    if file.endswith("*_oof_test.npy"):
        oof_test =np.load(file)
        #oof_test.reshape(-1,1)
        x_test=np.concatenate((x_test,oof_test),axis=1)
        del oof_test
        gc.collect()

#np.save('x_train', x_train)
#np.save('x_test', x_test)
print('train length',x_train.shape)
print('test length',x_test.shape)
print('y length',y_train.shape)

print("linear cv..")
lm=LinearRegression()
lm.fit(x_train,y_train)

ln_oof_test = lm.predict(x_test)


subm=pd.read_csv('../output/submission/lgb_m4.csv')
subm['PredictedLogRevenue'] = ln_oof_test
subm.to_csv('../output/submission/stack1_diff_data.csv', index=False)
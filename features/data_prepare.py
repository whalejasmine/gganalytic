#######module######

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import sys
import gc
#import seaborn as sns
#color = sns.color_palette()



##########prepare dataset##############
def load_df(filepath, csv_name='train.csv', nrows=None):
    if filepath is None:
        filepath='../input/'
    else:
        filepath=filepath
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(filepath+csv_name, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(filepath+csv_name)}. Shape: {df.shape}")
    return df

def load_post_df(filepath,csv_name='extracted_fields_train.gz',nrows=None):
    if filepath is None:
        filepath='../input/'
    else:
        filepath=filepath
    df = pd.read_csv(filepath+csv_name, dtype={'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': np.int64},nrows=nrows)
    print(f"Loaded {os.path.basename(filepath+csv_name)}. Shape: {df.shape}")
    return df
##########load & prepare external dataset##############
def load_ex_df(filepath,nrows=None):
    if filepath is None:
        filepath='../input/'

    # Getting data from leak
    train_store_1 = pd.read_csv(filepath+'Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'}, nrows=nrows)
    train_store_2 = pd.read_csv(filepath+'Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'}, nrows=nrows)
    test_store_1 = pd.read_csv(filepath+'Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'}, nrows=nrows)
    test_store_2 = pd.read_csv(filepath+'Test_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'}, nrows=nrows)
    # Clean data
    for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
        df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(np.int64)
        df.drop("Client Id", 1, inplace=True)
        df["Revenue"].fillna('$', inplace=True)
        df["Revenue"] = df["Revenue"].apply(lambda x: x.replace('$', '').replace(',', ''))
        df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
        df["Revenue"].fillna(0.0, inplace=True)
        df["Avg. Session Duration"][df["Avg. Session Duration"] == 0] = "00:00:00"
        df["Avg. Session Duration"] = df["Avg. Session Duration"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
        df["Bounce Rate"] = df["Bounce Rate"].astype(str).apply(lambda x: x.replace('%', '')).astype(float)
        df["Goal Conversion Rate"] = df["Goal Conversion Rate"].astype(str).apply(lambda x: x.replace('%', '')).astype(float)
    
    train_store=pd.concat([train_store_1, train_store_2], sort=False)
    test_store=pd.concat([test_store_1, test_store_2], sort=False)

    del train_store_1, train_store_2, test_store_1, test_store_2
    gc.collect()
    return train_store, test_store


##########define folding strategy##############
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids



##########impute missing##############


def imputmissing(train,test):
    y_reg = train['totals.transactionRevenue'].fillna(0)

    del train['totals.transactionRevenue']

    if 'totals.transactionRevenue' in test.columns:
        del test['totals.transactionRevenue']
    train['target']=y_reg
    return train, test, y_reg






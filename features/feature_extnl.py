import pandas as pd
import data_prepare
import time 
import warnings
import sys
import gc
import numpy as np
from contextlib import contextmanager
import logging
from pandas.core.common import SettingWithCopyWarning

warnings.filterwarnings("ignore")

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def session(df):
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df["date"] = df.visitStartTime



    df['sess_date_dow'] = df['date'].dt.dayofweek.astype(object)
    df['sess_date_time'] =  df['date'].dt.second + df['date'].dt.minute*60 + df['date'].dt.hour*3600
    df['sess_date_dom'] = df['date'].dt.day

    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60


    df.set_index("visitStartTime", inplace=True)
    df.sort_index(inplace=True)
    return df


##########start clear features##############
def clearRare(df,columnname, limit = 1000):
    # you may search for rare categories in train, train&test, or just test
    #vc = pd.concat([train[columnname], test[columnname]], sort=False).value_counts()
    vc = df[df['totals.transactionRevenue'].isnull()][columnname].value_counts()
    
    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, "categories to 'other';", end=" ")
    
    df.loc[df[columnname].map(lambda x: x not in common), columnname] = 'other'
    print("now there are",df[~df['totals.transactionRevenue'].isnull()][columnname].nunique(), "categories in train")




def add_features(df):

    print("mapping..")
    clearRare(df,"device.browser")
    clearRare(df,"device.operatingSystem")
    clearRare(df,"geoNetwork.country")
    clearRare(df,"geoNetwork.city")
    clearRare(df,"geoNetwork.metro")
    clearRare(df,"geoNetwork.networkDomain")
    clearRare(df,"geoNetwork.region")
    clearRare(df,"geoNetwork.subContinent")
    clearRare(df,"trafficSource.adContent")
    clearRare(df,"trafficSource.campaign")
    clearRare(df,"trafficSource.keyword")
    clearRare(df,"trafficSource.medium")
    clearRare(df,"trafficSource.referralPath")
    clearRare(df,"trafficSource.source")
    
    print("add strange phonomenan flag..")
    # remember these features were equal, but not always? May be it means something...
    df["id_incoherence"] = pd.to_datetime(df.visitId, unit='s') != df.date
    # remember visitId dublicates?
    df["visitId_dublicates"] = df.visitId.map(df.visitId.value_counts())
    # remember session dublicates?
    df["session_dublicates"] = df.sessionId.map(df.sessionId.value_counts())
    

    print("process..")
    df['source.country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    df['campaign.medium'] = df['trafficSource.campaign'] + '_' + df['trafficSource.medium']
    df['browser.category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser.os'] = df['device.browser'] + '_' + df['device.operatingSystem']

    print("custom..")
    df['device_deviceCategory_channelGrouping'] = df['device.deviceCategory'] + "_" + df['channelGrouping']
    df['channelGrouping_browser'] = df['device.browser'] + "_" + df['channelGrouping']
    df['channelGrouping_OS'] = df['device.operatingSystem'] + "_" + df['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            df[i + "_" + j] = df[i] + "_" + df[j]
    
    df['content.source'] = df['trafficSource.adContent'] + "_" + df['source.country']
    df['medium.source'] = df['trafficSource.medium'] + "_" + df['source.country']
    

    print("User-agg..")
    for feature in ["totals.hits", "totals.pageviews"]:
        info = df.groupby("fullVisitorId")[feature].mean()
        df["usermean_" + feature] = df.fullVisitorId.map(info)
    
    for feature in ["visitNumber"]:
        info = df.groupby("fullVisitorId")[feature].max()
        df["usermax_" + feature] = df.fullVisitorId.map(info)

    return df


def main(outpath,name,log,comment):

    if outpath is None:
        outpath='../output/feats/'
    if log is None:
        logfile = open('../output/log/{}.txt'.format(name), 'w')
    else:
        logfile = open('../output/log/{}.txt'.format(log), 'w')
    if comment is not None:
        logfile.write('comment: {}\n'.format(comment))    
    with timer("load data.."):
        train=data_prepare.load_post_df(None,'extracted_fields_train.gz')
        test=data_prepare.load_post_df(None, 'extracted_fields_test.gz')
        train_store, test_store=data_prepare.load_ex_df(filepath=None)
        # Merge with train/test data
        train = train.merge(train_store, how="left", on="visitId")
        train.fillna(0,inplace=True)
        test = test.merge(test_store, how="left", on="visitId")
        test.loc[:, test.columns.difference(['totals.transactionRevenue'])].fillna(0,inplace=True)
        df = pd.concat([train,test],axis=0,ignore_index=True)
        #df.fillna(0,inplace=True)

        logfile.write('df: train & test shape is: {}\n'.format(df.shape))
        del train, test
        gc.collect()

    with timer("Start feature engineering.."):

        df = session(df)
        logfile.write('after add session df shape:{}'.format(df.shape))

    with timer("add mapping features.."):

        df = add_features(df)
        logfile.write('after add mapping df shape:{}'.format(df.shape))

    with timer('finished generating features..'):
        df.head(100).to_csv(outpath+'df_sample.csv', index=False)
        df.reset_index(inplace=True)
        df.to_feather(outpath+name)




if __name__ == '__main__':

    main(None,'df_feats2.f',None,'https://www.kaggle.com/karkun/sergey-ivanov-msu-mmp')




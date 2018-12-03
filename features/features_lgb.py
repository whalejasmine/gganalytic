import gc
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

import warnings
import sys
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/ggalc/features') #import my module
import feature_cre
import data_prepare
import feather



excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]


def prep_and_split(df):

    
    ##########factortized categoricals##############
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False)==1 ]
    print('constant conlumns is:',const_cols)

    df.drop(const_cols, axis=1, inplace=True)

    categorical_features = [_f for _f in df.columns if (_f not in excluded_features) & (df[_f].dtype == 'object')]
    for f in categorical_features:
        df[f], indexer = pd.factorize(df[f])

    X_train = df[df['tn']==1].drop('totals.transactionRevenue', axis=1)
    y_train = df[df['tn']==1]['totals.transactionRevenue']
    X_test = df[ddf['tn']==0].drop('totals.transactionRevenue', axis=1)
    print('X_train shape:',X_train.shape)
    print('y_train shape:',y_train.shape)
    print('X_test shape:',X_test.shape)

    return X_train, y_train, X_test


def add_pred_feats(df,preds):
    df['predictions'] =  np.expm1(preds)
    # Aggregate data at User level
    trn_data = df.groupby('fullVisitorId').mean()

    # Create a list of predictions for each Visitor
    trn_pred_list = df[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
        .apply(lambda df: list(df.predictions))\
        .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
    trn_feats = trn_all_predictions.columns
    trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
    trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
    trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
    trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
    trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
    full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
    del trn_data, trn_all_predictions
    gc.collect()
    print('Shape of data after add predictions feats:',full_data.shape)

    return full_data

# Baseline CV:0.7950 LB:0.798
class lightgbm(object):
    def __init__(self, name, comment=None, remove_columns = None,
                 param = None, lgb_seed = None, 
                 n_estimators = 1000, log = None,
                 predict_feats=False, debug=True, BASE_X_PATH=None):
        self.name = name
        self.comment = comment

        if log is None:
            self.logfile = open('../output/log/{}.txt'.format(name), 'w')
        else:
            self.logfile = open('../output/log/{}.txt'.format(log), 'w')

        if param is None:
            self.param = {

                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'goss',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'n_jobs': -1,
                'random_state': 456,
                #'verbose': 100,
            }
        else:
            self.param = param

        if lgb_seed is not None:
            self.param['seed'] = lgb_seed
        self.param['n_estimators'] = n_estimators
        self.feature_importance_df = None
        self.regressors = []
        
        if BASE_X_PATH is not None:
            self.x = pd.read_csv(BASE_X_PATH)
        
        if remove_columns is not None:
            drop_features =[_f for _f in self.x.columns if _f in remove_columns]
            self.x.drop(drop_features, axis=1, inplace=True)
            del drop_features
            gc.collect()

        #read & prepare datasets
        print('read & prepare datasets shape: {}'.format(self.x.shape))

        #split train & test sets
        self.x_train, self.y_train, self.x_test = prep_and_split(self.x)

        #debug
        if debug:
            x_train_s=self.x_train.sample(frac=0.3)
            x_test_s=self.x_train.sample(frac=0.3)
            y_train_s=self.y_train.loc[self.y_train.index.isin(x_train_s.index)]


        if predict_feats:
            self.x_train = add_pred_feats(x_train_s,oof_preds)
            self.x_test = add_pred_feats(x_test_s,preds_test)
            self.y_train = y_train_s.reset_index().groupby('fullVisitorId').sum()
            del x_train_s,x_test_s,y_train_s
            gc.collect()
        else:
            self.x_train.reset_index(drop=True,inplace=True)
            self.x_test.reset_index(drop=True, inplace=True)


    # Display/plot feature importance
    def display_importances(self, feature_importance_df_, filename, n=100):
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                       ascending=False)[:n].index

        import matplotlib.pyplot as plt
        #import seaborn as sns

        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        best_features.to_excel('../output/{}.xlsx'.format(filename))
        #plt.figure(figsize=(10, 16))
        #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        #plt.title('LightGBM Features (avg over folds)')
        #plt.tight_layout()
        #plt.savefig('{}.png'.format(filename))

    def cv(self, nfolds=5, submission=True):
        self.regressors.clear()
        self.feature_importance_df = pd.DataFrame()

        if not submission:
            folds = data_prepare.get_folds(df=self.x_train, n_splits = nfolds)
        else:
            folds = data_prepare.get_folds(df=self.x_train[['totals.pageviews']].reset_index(), n_splits = nfolds)

        
        if 'fullVisitorId' in self.x_train.columns:
            self.x_train.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.x_test.columns:
            self.x_test.drop('fullVisitorId', axis=1, inplace=True)
        #if 'fullVisitorId' in self.y_train.columns:
            #self.y_train.drop('fullVisitorId', axis=1, inplace=True)



        oof_preds = np.zeros(self.x_train.shape[0])
        preds_test = np.empty((nfolds, self.x_test.shape[0]))



        self.logfile.write('param: {}\n'.format(self.param))
        self.logfile.write('fold: {}\n'.format(nfolds))
        self.logfile.write('data shape: {}\n'.format(self.x_train.shape))
        self.logfile.write('features: {}\n'.format(self.x_train.columns.tolist()))
        
        if self.comment is not None:
            self.logfile.write('comment: {}\n'.format(self.comment))

        self.logfile.write('output: ../output/{}.csv\n'.format(self.name))
        self.logfile.flush()

        for n_fold, (train_idx, valid_idx) in enumerate(folds):
            fstart = time.time()
            train_x, train_y = self.x_train.iloc[train_idx], self.y_train.iloc[train_idx]
            valid_x, valid_y = self.x_train.iloc[valid_idx], self.y_train.iloc[valid_idx]

            # lgbRegressor parameters found by Bayesian optimization
            clf = LGBMRegressor(**self.param)
            clf.fit(train_x, np.log1p(train_y), eval_set=[(valid_x, np.log1p(valid_y))],
                    eval_metric='rmse', verbose=100, early_stopping_rounds=200)

            oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
            preds_test[n_fold, :] = clf.predict(self.x_test, num_iteration=clf.best_iteration_)

            #remove negative and transform un log
            oof_preds[oof_preds<0]=0
            preds_test[preds_test<0]=0

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = self.x_train.columns.tolist()
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

            strlog = '[{}][{:.1f} sec] Fold {} RMSE : {:.6f}'.format(str(datetime.now()), time.time() - fstart, n_fold + 1, mean_squared_error(np.log1p(valid_y), oof_preds[valid_idx]) ** .5 )
            print(strlog)
            self.logfile.write(strlog+'\n')
            self.logfile.flush()

            self.regressors.append(clf)
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        full_rmse = mean_squared_error(np.log1p(self.y_train), oof_preds) ** .5 
        strlog = 'Full RMSE score {:.6f}'.format(full_rmse)
        print(strlog)
        self.logfile.write(strlog+'\n')
        
        preds= preds_test.mean(axis=0)

        if submission:
            #sub = pd.read_csv('../input/sample_submission.csv')
            #sub['PredictedLogRevenue'] = preds
            preds.to_csv('../output/submission/{}.csv'.format(self.name), index=True)

            cols = self.feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:100].index
            self.logfile.write('top features:\n')
            for c in cols:
                self.logfile.write('{}\n'.format(c))

            self.logfile.flush()

            self.display_importances(self.feature_importance_df, self.name)
            
        # for stack
        np.save('../output/feats/{}_trn_prd_feats'.format(self.name),oof_preds)
        np.save('../output/feats/{}_tes_prd_feats'.format(self.name),preds)
        return self.feature_importance_df, full_rmse, oof_preds, preds



if __name__ == "__main__":
    predfts = lightgbm(name='lgb_predfts',remove_columns = excluded_features, comment='lgb_predfts',predict_feats=False, debug=False, BASE_X_PATH = '../output/feats/df_feats1.f')
    df_feats_revn, full_rmse, oof_preds, preds_test = predfts.cv(submission=False)


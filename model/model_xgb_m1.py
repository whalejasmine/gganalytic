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
sys.path.insert(0, '../features') #import my module
import feature_cre
import data_prepare
import feather
from bayes_optz import BayesianOptimization


excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]


def prep_and_split(df):

    
    ##########factortized categoricals##############
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False)==1 ]
    print('constant conlumns is:',const_cols)

    df.drop(const_cols, axis=1, inplace=True)

    categorical_features = [_f for _f in df.columns if (_f not in excluded_features+['fullVisitorId']) & (df[_f].dtype == 'object')]
    for f in categorical_features:
        df[f], indexer = pd.factorize(df[f])

    X_train = df[~df['totals.transactionRevenue'].isnull()].drop('totals.transactionRevenue', axis=1)
    y_train = df[~df['totals.transactionRevenue'].isnull()]['totals.transactionRevenue']
    y_train_ag=df[~df['totals.transactionRevenue'].isnull()][['totals.transactionRevenue','fullVisitorId']]
    X_test = df[df['totals.transactionRevenue'].isnull()].drop('totals.transactionRevenue', axis=1)
    print('X_train shape:',X_train.shape)
    print('y_train shape:',y_train.shape)
    print('X_test shape:',X_test.shape)
    print('y_train_ag shape:',y_train_ag.shape)
    del df
    gc.collect()
    return X_train, y_train, X_test, y_train_ag


def add_pred_feats(df,preds,trn_feats=None):
    print('df shape:',df.shape)
    print('preds:',preds.shape)
    df['predictions'] =  np.expm1(preds)
    # Aggregate data at User level
    trn_data = df.groupby('fullVisitorId').mean()
    trn_data_sm = df.groupby('fullVisitorId').sum().add_suffix('_sum')
 #   trn_data_min = df.groupby('fullVisitorId').min().add_suffix('_min')
 #   trn_data_max = df.groupby('fullVisitorId').max().add_suffix('_max')
    trn_data_std = df.groupby('fullVisitorId').std().add_suffix('_std')
 
    # Create a list of predictions for each Visitor
    trn_pred_list = df[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
        .apply(lambda df: list(df.predictions))\
        .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
    
    if trn_feats is None:
        trn_feats = trn_all_predictions.columns
    else:
        for f in trn_feats:
            if f not in trn_all_predictions.columns:
                trn_all_predictions[f]=np.nan
    trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
    trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
    trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
    trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
    trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
    full_data = pd.concat([trn_data,trn_data_sm,
                            #trn_data_min,trn_data_max,
                            trn_data_std, trn_all_predictions], axis=1)
    del trn_data, trn_all_predictions
    gc.collect()
    print('Shape of data after add predictions feats:',full_data.shape)

    return full_data, trn_feats





# Baseline CV:0.7950 LB:0.798
class xgbl(object):
    def __init__(self, name, clf,comment=None, remove_columns = None,
                 param = None, xgb_seed = None, 
                 n_estimators = 1000, log = None,
                 predict_feats=False, debug=True
                 , BASE_X_PATH=None, TRN_PRED_FEATS=None, TES_PRED_FEATS=None):
        self.name = name
        self.comment = comment

        if log is None:
            self.logfile = open('../output/log/{}.txt'.format(name), 'w')
        else:
            self.logfile = open('../output/log/{}.txt'.format(log), 'w')

        if param is None:
            self.param = {

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
                'sead':6
                #'verbose': 100,
            }
        else:
            self.param = param

        if xgb_seed is not None:
            self.param['seed'] = xgb_seed
        self.param['n_estimators'] = n_estimators
        self.feature_importance_df = None

        self.Tunparams=None
        if BASE_X_PATH is not None:
            self.x = feather.read_dataframe(BASE_X_PATH)
        if TRN_PRED_FEATS is not None:
            self.trn_preds_feats=np.load(TRN_PRED_FEATS)
        if TES_PRED_FEATS is not None:
                self.tes_preds_feats=np.load(TES_PRED_FEATS)

        if remove_columns is not None:
            drop_features =[_f for _f in self.x.columns if _f in remove_columns]
            self.x.drop(drop_features, axis=1, inplace=True)
            del drop_features
            gc.collect()
        self.clf = clf(**self.param)
        #read & prepare datasets
        print('read & prepare datasets shape: {}'.format(self.x.shape))

        #split train & test sets
        self.x_train, self.y_train, self.x_test, self.y_train_ag = prep_and_split(self.x)

        #debug
        if debug:
            x_train_s=self.x_train.sample(frac=0.3)
            x_test_s=self.x_test.sample(frac=0.3)
            y_train_s=self.y_train_ag.loc[self.y_train_ag.index.isin(x_train_s.index)]
        else:
            x_train_s=self.x_train.sample(frac=1)
            x_test_s=self.x_test.sample(frac=1)
            y_train_s=self.y_train_ag.loc[self.y_train_ag.index.isin(x_train_s.index)]            

        if predict_feats:
            self.x_train,trn_feats = add_pred_feats(x_train_s,self.trn_preds_feats, None)
            self.x_test,_ = add_pred_feats(x_test_s,self.tes_preds_feats,trn_feats)
            self.y_train = y_train_s.groupby('fullVisitorId').sum()
            del x_train_s,x_test_s,y_train_s
            gc.collect()
        else:
            self.x_train.reset_index(drop=True,inplace=True)
            self.x_test.reset_index(drop=True, inplace=True)


    def xgb_evaluate(self,**Tunparams):
        warnings.filterwarnings("ignore")
        
        Tunparams['max_depth'] = int(Tunparams['max_depth'])
        Tunparams['min_child_weight'] = int(Tunparams['min_child_weight'])
        Tunparams['cosample_bytree'] = max(min(Tunparams['colsample_bytree'], 1), 0)
        Tunparams['subsample'] = max(min(Tunparams['subsample'], 1), 0)
        Tunparams['gamma'] = max(Tunparams['gamma'], 0)
        Tunparams['alpha'] = max(Tunparams['alpha'], 0)
        folds = data_prepare.get_folds(df=self.x_train[['totals.pageviews']].reset_index(), n_splits = 2)

        
        if 'fullVisitorId' in self.x_train.columns:
            self.x_train.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.x_test.columns:
            self.x_test.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.y_train.columns:
            self.y_train.drop('fullVisitorId', axis=1, inplace=True)
        
            
        oof_preds = np.zeros(self.x_train.shape[0])
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds):
            train_x, train_y = self.x_train.iloc[train_idx], self.y_train.iloc[train_idx]
            valid_x, valid_y = self.x_train.iloc[valid_idx], self.y_train.iloc[valid_idx]

            # LGBMRegressor parameters found by Bayesian optimization
            self.clf.fit(train_x, np.log1p(train_y), eval_set=[(valid_x, np.log1p(valid_y))],
                    eval_metric='rmse', verbose=100, early_stopping_rounds=200)

            oof_preds[valid_idx] = self.clf.predict(valid_x, ntree_limit=self.clf.best_iteration)

            #remove negative and transform un log
            oof_preds[oof_preds<0]=0
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        return -mean_squared_error(np.log1p(self.y_train), oof_preds) ** .5

    # Display/plot feature importance
    def display_importances(self, feature_importance_df_, filename, n=50):
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                       ascending=False)[:n].index

        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        import seaborn as sns

        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        best_features.to_excel('../output/{}.xlsx'.format(filename))
        plt.figure(figsize=(10, 16))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('Xgb Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('../output/{}.png'.format(filename))

    def cv(self, nfolds=5, submission=True,tune=True,nthread=16):
        self.feature_importance_df = pd.DataFrame()
        self.Tunparams={'colsample_bytree': (0.5, 1),
                          'learning_rate': (.01, .05), 
                          'subsample': (0.8, 1), 
                          'max_depth': (1, 10), 
                          'alpha': (.02, .10), 
                          'lambda': (.02, .08), 
                          'gamma':(0,10),
                          'min_child_weight': (1, 40)}

        folds = data_prepare.get_folds(df=self.x_train[['totals.pageviews']].reset_index(), n_splits = nfolds)

        
        if 'fullVisitorId' in self.x_train.columns:
            self.x_train.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.x_test.columns:
            self.x_test.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.y_train.columns:
            self.y_train.drop('fullVisitorId', axis=1, inplace=True)

        print('y_train columns:',self.y_train.columns)
        print('y_train shape drop id?:',self.y_train.shape)
        np.where((self.x_test.applymap(type)==object))



        oof_preds = np.zeros(self.x_train.shape[0])
        preds_test = np.empty((nfolds, self.x_test.shape[0]))

        #Tune parameters
        if tune:
                bo = BayesianOptimization(self.xgb_evaluate, self.Tunparams)
                bo.maximize(init_points = 5, n_iter =50,acq='rnd')


                best_params = bo.res['max']['max_params']
                best_params['num_leaves'] = int(best_params['num_leaves'])
                best_params['max_depth'] = int(best_params['max_depth'])
                self.logfile.write('best param process: {}\n'.format(bo.res['max']['max_val']))
                self.param.update(best_params)
                self.logfile.write('best param: {}\n'.format(self.param))

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

            # XGBRegressor parameters found by Bayesian optimization
            self.clf.fit(train_x, np.log1p(train_y), eval_set=[(valid_x, np.log1p(valid_y))],
                    eval_metric='rmse', verbose=100, early_stopping_rounds=200)

            oof_preds[valid_idx] = self.clf.predict(valid_x, ntree_limit=self.clf.best_iteration)
            preds_test[n_fold, :] = self.clf.predict(self.x_test[self.x_train.columns], ntree_limit=self.clf.best_iteration)

            #remove negative and transform un log
            oof_preds[oof_preds<0]=0
            preds_test[preds_test<0]=0

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = self.x_train.columns.tolist()
            fold_importance_df["importance"] = self.clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

            strlog = '[{}][{:.1f} sec] Fold {} RMSE : {:.6f}'.format(str(datetime.now()), time.time() - fstart, n_fold + 1, mean_squared_error(np.log1p(valid_y), oof_preds[valid_idx]) ** .5 )
            print(strlog)
            self.logfile.write(strlog+'\n')
            self.logfile.flush()

            del  train_x, train_y, valid_x, valid_y
            gc.collect()

        full_rmse = mean_squared_error(np.log1p(self.y_train), oof_preds) ** .5 
        strlog = 'Full RMSE score {:.6f}'.format(full_rmse)
        print(strlog)
        self.logfile.write(strlog+'\n')
        
        preds= preds_test.mean(axis=0)
        if submission:
            #sub = pd.read_csv('./input/sample_submission.csv')
            #sub['PredictedLogRevenue'] = preds
            self.x_test['PredictedLogRevenue'] = preds
            self.x_test[['PredictedLogRevenue']].to_csv('../output/submission/{}.csv'.format(self.name), index=True)
            #preds.to_csv('../output/{}.csv'.format(self.name), index=True)

            cols = self.feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:20].index
            self.logfile.write('top features:\n')
            for c in cols:
                self.logfile.write('{}\n'.format(c))

            self.logfile.flush()

            self.display_importances(self.feature_importance_df, self.name)
            
        # for stack
        np.save('../output/log/{}_oof_train'.format(self.name),oof_preds)
        np.save('../output/log/{}_oof_test'.format(self.name),preds)
        return self.feature_importance_df, full_rmse, oof_preds, preds



if __name__ == "__main__":

    m = xgbl(name='xgbl_m1',remove_columns = excluded_features
            , comment='xgbl_m1',predict_feats=True,debug=False
            ,BASE_X_PATH = '../output/df_feats1.f'
            ,TRN_PRED_FEATS='../output/xgbl_predfts_trn_prd_feats.npy'
            ,TES_PRED_FEATS='../output/xgbl_predfts_tes_prd_feats.npy'
            ,clf=XGBRegressor)
    df_feats, full_rmse, _, _= m.cv(tune=False)

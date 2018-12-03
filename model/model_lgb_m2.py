import gc
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.insert(0, '../features') #import my module
import feature_cre
import data_prepare
import feather
from bayes_optz import BayesianOptimization
from lightgbm import LGBMRegressor



excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]


def prep_and_split(df):

    ##########constant features##############
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False)==1 ]
    print('constant conlumns is:',const_cols)
    df.drop(const_cols, axis=1, inplace=True)

    ##########factortized categoricals##############
    categorical_features = [_f for _f in df.columns if (_f not in excluded_features+['fullVisitorId']) & (df[_f].dtype == 'object')]
    for f in categorical_features:
        df[f], indexer = pd.factorize(df[f])

    X_train = df[~df['totals.transactionRevenue'].isnull()].drop('totals.transactionRevenue', axis=1)
    y_train = df[~df['totals.transactionRevenue'].isnull()]['totals.transactionRevenue']
    y_train_ag=df[~df['totals.transactionRevenue'].isnull()][['totals.transactionRevenue','fullVisitorId']]
    X_test = df[df['totals.transactionRevenue'].isnull()].drop('totals.transactionRevenue', axis=1)

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)


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
    print(full_data.reset_index().fullVisitorId.head())
    return full_data, trn_feats

# Baseline CV:0.7950 LB:0.798
class lightgbm(object):

    def __init__(self, name, clf,comment=None, remove_columns = None,
                 param = None, lgb_seed = None, 
                 n_estimators = 1000, log = None,
                 predict_feats=False, debug=True,
                 BASE_X_PATH=None, TRN_PRED_FEATS=None, TES_PRED_FEATS=None
                 ,nfolds=5):


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
                'num_leaves': 32,
                'learning_rate': 0.03,
                'colsample_bytree': 0.9,
                'n_estimators': 1000,
                'boosting_type': 'goss'
            }
        else:
            self.param = param

        if lgb_seed is not None:
            self.param['seed'] = lgb_seed
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

        self.clf=clf(**self.param)

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
       
        self.folds = data_prepare.get_folds(df=self.x_train[['totals.pageviews']].reset_index(), n_splits = nfolds)


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
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('../output/{}.png'.format(filename))

    def lgbm_evaluate(self,**Tunparams):
        warnings.filterwarnings("ignore")
        
        Tunparams['num_leaves'] = int(Tunparams['num_leaves'])
        Tunparams['max_depth'] = int(Tunparams['max_depth'])
        
        #folds = data_prepare.get_folds(df=self.x_train[['totals.pageviews']].reset_index(), n_splits = 2)

        
        if 'fullVisitorId' in self.x_train.columns:
            self.x_train.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.x_test.columns:
            self.x_test.drop('fullVisitorId', axis=1, inplace=True)
        if 'fullVisitorId' in self.y_train.columns:
            self.y_train.drop('fullVisitorId', axis=1, inplace=True)
            
        oof_preds = np.zeros(self.x_train.shape[0])
        
       
        for n_fold, (train_idx, valid_idx) in enumerate(self.folds):
            train_x, train_y = self.x_train.iloc[train_idx], self.y_train.iloc[train_idx]
            valid_x, valid_y = self.x_train.iloc[valid_idx], self.y_train.iloc[valid_idx]

            # LGBMRegressor parameters found by Bayesian optimization
            self.clf.fit(train_x, np.log1p(train_y), eval_set=[(valid_x, np.log1p(valid_y))],
                    eval_metric='rmse', verbose=100, early_stopping_rounds=200)

            oof_preds[valid_idx] = self.clf.predict(valid_x, num_iteration=self.clf.best_iteration_)

            #remove negative and transform un log
            oof_preds[oof_preds<0]=0
            del  train_x, train_y, valid_x, valid_y
            gc.collect()


        return -mean_squared_error(np.log1p(self.y_train), oof_preds) ** .5

    def cv(self, X, nfolds=5, submission=True, tune=True, baseline=None):
        
        rmse=[]
        n_lose = 0
        balance = 0

        if X is not None:
            self.x_train = X
            
        self.feature_importance_df = pd.DataFrame()
        self.Tunparams={'colsample_bytree': (0.3, 1),
                          'learning_rate': (.02, .1), 
                          'num_leaves': (20, 50), 
                          'subsample': (0.3, 1), 
                          'max_depth': (2, 6), 
                          'reg_alpha': (.03, .1), 
                          'reg_lambda': (.06, .1), 
                          'min_split_gain': (.01, .1),
                          'min_child_weight': (10, 50)
                          }


        
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
        preds=np.zeros((self.x_test.shape[0],))
        preds_test = np.empty((nfolds, self.x_test.shape[0]))

        print('oof_preds shape',oof_preds.shape)
        print('preds shape',preds.shape)
        print('preds_test shape',preds_test.shape)

        #Tune parameters
        if tune:
                bo = BayesianOptimization(self.lgbm_evaluate, self.Tunparams)
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
        
        if self.comment is not None:
            self.logfile.write('comment: {}\n'.format(self.comment))

        self.logfile.write('output: ../output/{}.csv\n'.format(self.name))
        self.logfile.flush()
        


        for n_fold, (train_idx, valid_idx) in enumerate(self.folds):
            fstart = time.time()
            train_x, train_y = self.x_train.iloc[train_idx], self.y_train.iloc[train_idx]
            valid_x, valid_y = self.x_train.iloc[valid_idx], self.y_train.iloc[valid_idx]

            # LGBMRegressor parameters found by Bayesian optimization
            self.clf.fit(train_x, np.log1p(train_y), eval_set=[(valid_x, np.log1p(valid_y))],
                    eval_metric='rmse', verbose=100, early_stopping_rounds=200)

            oof_preds[valid_idx] = self.clf.predict(valid_x, num_iteration=self.clf.best_iteration_)

            if submission:
                preds_test[n_fold, :] =self.clf.predict(self.x_test, num_iteration=self.clf.best_iteration_)

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


            #fold rmse
            fold_rmse= mean_squared_error(np.log1p(valid_y), oof_preds[valid_idx]) ** .5
            if baseline is not None:
                balance = balance + fold_rmse - baseline[n_fold]
                if fold_rmse < baseline[n_fold]:
                    n_lose = n_lose + 1

            rmse.append(fold_rmse)                


            if baseline is not None and n_lose > nfolds/2 and balance < 0:
                return rmse

            del train_x, train_y, valid_x, valid_y
            gc.collect()

        full_rmse = mean_squared_error(np.log1p(self.y_train), oof_preds) ** .5 
        rmse.append(full_rmse)  

        strlog = 'Full RMSE score {:.6f}'.format(full_rmse)
        print(strlog)
        self.logfile.write(strlog+'\n')
        
        preds[:]= preds_test.mean(axis=0)
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
            np.save('../output/log/{}_y_train'.format(self.name),self.y_train)
        return self.feature_importance_df, full_rmse, oof_preds, preds, rmse


    def feature_selection_eval(self, nfolds, set=2, file='../output/log/log_fw.txt', earlystop=True):
        n_columns = self.x_train.shape[1]

        n_loop = n_columns // set

        with open(file, 'a') as f:
            # baseline. 
            _,_,_,_,rmse = self.cv(None,5,False,False,None)
            f.write('{},{},{},{},{},{},baseline,baseline\n'.format(rmse[0], rmse[1], rmse[2], rmse[3], rmse[4], rmse[5]))

            if earlystop:
                self.baseline = rmse
            else:
                self.baseline = None

            for i in range(n_loop):
                X_c = self.x_train.reset_index().copy()
                for n in range(set):
                    idx = i * set + n
                    col = self.x_train.columns.tolist()[idx]
                    X_c[col] = self.x_train[col]

                add_columns = self.x_train.columns.tolist()[i * set:(i + 1) * set]
                print('add:{}'.format(add_columns))
                _,_,_,_,rmse = self.cv(X_c,5,False,False, self.baseline)

                for a in rmse:
                    f.write('{},'.format(a))
                for a in add_columns:
                    f.write('{},'.format(a))
                f.write('\n')
                f.flush()



if __name__ == "__main__":

    m = lightgbm(name='lgb_m2_test',remove_columns = excluded_features, comment='lgb model using feats_xgb with test',predict_feats=True,debug=False,lgb_seed=17,
                    BASE_X_PATH='../output/feats/df_feats2.f', TRN_PRED_FEATS='../output/feats/xgbl_predfts_2_trn_prd_feats.npy',
                    TES_PRED_FEATS='../output/feats/xgbl_predfts_2_tes_prd_feats.npy',clf=LGBMRegressor)
    df_feats, full_rmse, _, _= m.cv(tune=True)

import model_lgb_m2
from lightgbm import LGBMRegressor

class lgbm_tun(model_lgb_m2.lightgbm):
    def __init__(self):

        super().__init__(name='lgb_m6', clf=LGBMRegressor,comment='lgb_predfts add mean medium sum std feature of visitor level in cats datasets'
                        , BASE_X_PATH = '../output/feats/df_feats_cats.f',remove_columns = excluded_features
                        , TRN_PRED_FEATS='../output/feats/lgb_predfts_2_trn_prd_feats.npy'
                        ,TES_PRED_FEATS='../output/feats/lgb_predfts_2_tes_prd_feats.npy'
                        ,predict_feats=True,debug=False,lgb_seed=17
                        ,param={'colsample_bytree': 0.592169, 
                                'learning_rate': 0.070698, 
                                'num_leaves': 33, 
                                'subsample': 0.8849, 
                                'max_depth': 3, 
                                'reg_alpha': 0.039149, 
                                'reg_lambda': 0.06007, 
                                'min_split_gain': 0.060853, 
                                'min_child_weight': 37.22})

if __name__ == "__main__":
    excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]

    m = lgbm_tun()

    m.cv(X=None,tune=False)
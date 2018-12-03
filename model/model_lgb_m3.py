import model_lgb_m2
from lightgbm import LGBMRegressor

class lgbm_tun(model_lgb_m2.lightgbm):
    def __init__(self):

        super().__init__(name='lgb_predfts_t2', clf=LGBMRegressor,comment='lgb_predfts with external tune with lgb feats'
                        , BASE_X_PATH = '../output/feats/df_feats2.f',remove_columns = excluded_features
                        , TRN_PRED_FEATS='../output/feats/lgb_predfts_2_trn_prd_feats.npy'
                        ,TES_PRED_FEATS='../output/feats/lgb_predfts_2_tes_prd_feats.npy'
                        ,predict_feats=True,debug=False,lgb_seed=17)

if __name__ == "__main__":
    excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]

    m = lgbm_tun()

    m.cv(tune=True)
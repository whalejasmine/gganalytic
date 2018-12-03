import model_xgb_m1
from xgboost import XGBRegressor

class xgbm_tun(model_xgb_m1.xgbl):
    def __init__(self):

        super().__init__(name='xgb_m6', comment='xgb_predfts with external not tune with xgb feats based on t1'
                        , BASE_X_PATH = '../output/feats/df_feats2.f',remove_columns = excluded_features
                        , TRN_PRED_FEATS='../output/feats/xgbl_predfts_3_trn_prd_feats.npy'
                    	,TES_PRED_FEATS='../output/feats/xgbl_predfts_3_tes_prd_feats.npy'
                        ,predict_feats=True,debug=False,xgb_seed=17
                        ,clf=XGBRegressor)

if __name__ == "__main__":
    excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index']

    m = xgbm_tun()

    m.cv(tune=False)
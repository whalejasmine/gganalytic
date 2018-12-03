import model_lm_m
from sklearn.linear_model import Lasso

class lasso(model_lm_m.lm):
    def __init__(self):

        super().__init__(name='lm_m4', clf=Lasso,comment='lasso_m1 with cat '
                        , BASE_X_PATH = '../output/feats/df_feats_cats.f',remove_columns = excluded_features
                        , TRN_PRED_FEATS='../output/feats/lm_feats_trn_prd_feats.npy'
                        ,TES_PRED_FEATS='../output/feats/lm_feats_tes_prd_feats.npy'
                        ,predict_feats=False,debug=False,lm_seed=17
                        ,param={'alpha': 0.51932
                                }
                        , scale=True)

if __name__ == "__main__":
    excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]

    m = lasso()

    m.cv(tune=True)
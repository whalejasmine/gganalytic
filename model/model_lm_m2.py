import model_lm_m
from sklearn.linear_model import Ridge

class lgbm_tun(model_lm_m.lm):
    def __init__(self):

        super().__init__(name='lm_m2', clf=Ridge,comment='LM with cat and feature sealect backward'
                        , BASE_X_PATH = '../output/feats/df_feats_cats.f',remove_columns = excluded_features
                        , TRN_PRED_FEATS='../output/feats/lm_feats_trn_prd_feats.npy'
                        ,TES_PRED_FEATS='../output/feats/lm_feats_tes_prd_feats.npy'
                        ,predict_feats=False,debug=False,lm_seed=17
                        ,param={'alpha': 0.51932
                                })

if __name__ == "__main__":
    excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]

    m = lgbm_tun()
    import sys
    argc = len(sys.argv)

    filename = '../output/log/log_fw_lm.txt' if argc == 1 else sys.argv[1]
    nset = 5 if argc < 3 else int(sys.argv[2])
    nskip = 0 if argc < 4 else int(sys.argv[3])

print('file:{}, n-set:{}, n-skip:{}'.format(filename, nset, nskip))
m.feature_selection_eval(nfolds=5, set=1, file=filename,nskip=nskip)
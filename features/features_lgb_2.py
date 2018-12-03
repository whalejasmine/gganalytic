

import features_lgb
class f_lgbm_2(features_lgb.lightgbm):
    def __init__(self):

        super().__init__(name='lgb_predfts_v2', comment='lgb_predfts with external'
                        , BASE_X_PATH = '../output/feats/df_feats_v2.csv',remove_columns = excluded_features)

if __name__ == "__main__":
    excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]

    m = f_lgbm_2()

    m.cv(submission=False)
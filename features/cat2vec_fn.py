import pandas as pd
import numpy as np
import gc, copy
from gensim.models import Word2Vec # categorical feature to vectors
from random import shuffle
import warnings
import feather
warnings.filterwarnings('ignore')
BASE_X_PATH = '../output/feats/df_feats_v2.csv'
excluded_features = [ #'fullVisitorId', #exclude this first for group folds and drop
'date',  'sessionId',  
'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','index'
]
class cat2vec_tf(object):
    def __init__(self, name, comment=None):
        self.name = name
        self.comment = comment

        self.df =pd.read_csv(BASE_X_PATH)
        self.tr_index=self.df[~self.df['totals.transactionRevenue'].isnull()].index
        self.te_index=self.df[self.df['totals.transactionRevenue'].isnull()].index

        self.cat_cols = [_f for _f in self.df.columns if (_f not in excluded_features+['fullVisitorId']) & (self.df[_f].dtype == 'object')]
        self.n_cat2vec_feature  = len(self.cat_cols) # define the cat2vecs dimentions
        self.n_cat2vec_window   = len(self.cat_cols) * 2 # define the w2v window size
        
        print('Cat features:',self.n_cat2vec_feature)

    #Define a help function for transform sentences to vectors with a w2v model
    def apply_w2v(self,sentences, model, num_features):
        def _average_word_vectors(words, model, vocabulary, num_features):
            feature_vector = np.zeros((num_features,), dtype="float64")
            n_words = 0.
            for word in words:
                if word in vocabulary: 
                    n_words = n_words + 1.
                    feature_vector = np.add(feature_vector, model[word])

            if n_words:
                feature_vector = np.divide(feature_vector, n_words)
            return feature_vector
        
        vocab = set(model.wv.index2word)
        feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
        return np.array(feats)

    #Define a function for generating category sentences¶
    def gen_cat2vec_sentences(self,data):
        X_w2v = copy.deepcopy(data)
        names = list(X_w2v.columns.values)
        for c in names:
            X_w2v[c] = X_w2v[c].fillna('unknow').astype('category')
            X_w2v[c].cat.categories = ["%s %s" % (c,g) for g in X_w2v[c].cat.categories]
        X_w2v = X_w2v.values.tolist()
        return X_w2v

    def fit_cat2vec_model(self):
        X_w2v = self.gen_cat2vec_sentences(self.df.loc[:,self.cat_cols].sample(frac=0.6))
        for i in X_w2v:
            shuffle(i)
        model = Word2Vec(X_w2v, size= self.n_cat2vec_feature, window=self.n_cat2vec_window)
        return model

    def get_vec(self):

        c2v_model = self.fit_cat2vec_model()
        tr_c2v_matrix = self.apply_w2v(self.gen_cat2vec_sentences(self.df.loc[self.tr_index,self.cat_cols]), c2v_model, self.n_cat2vec_feature)
        te_c2v_matrix = self.apply_w2v(self.gen_cat2vec_sentences(self.df.loc[self.te_index,self.cat_cols]), c2v_model, self.n_cat2vec_feature)
        print('tr_c2v_matrix shape:',tr_c2v_matrix.shape)
        print('te_c2v_matrix shape:',te_c2v_matrix.shape)
        self.df.loc[self.tr_index,self.cat_cols]=tr_c2v_matrix
        self.df.loc[self.te_index,self.cat_cols]=te_c2v_matrix
        self.df.head(100).to_csv('../output/feats/df_sample_cats.csv', index=False)
        self.df.reset_index(inplace=True)
        self.df.to_feather('../output/feats/'+self.name)

if __name__ == "__main__":
    catv=cat2vec_tf('df_feats_cats_v2.f')
    catv.get_vec()



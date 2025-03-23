import numpy as np
import pandas as pd
class TfIdfVectorizer:
    def __init__(self,tf_idf,uniq_toks,v_size):
        self.tf_idf = tf_idf
        self.uniq_toks = uniq_toks
        self.size = v_size
        
    def _toVector(self, iid):
        v = np.zeros(self.size)
        idxs = self.uniq_toks[iid]
        if idxs.size == 0:
            return v
        v[idxs] = self.tf_idf[iid]
        return v
    
    def toVector(self, iids):
        if type(iids) == str:
            return self._toVector(iids)
        else:
            vs = [self._toVector(iid) for iid in iids]
            return pd.Series(vs, index=iids)
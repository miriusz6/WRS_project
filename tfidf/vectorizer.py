import numpy as np

class TfIdfVectorizer:
    def __init__(self,tf_idf,uniq_toks,v_size):
        self.tf_idf = tf_idf
        self.uniq_toks = uniq_toks
        self.size = v_size
        
    def toVector(self, iid):
        v = np.zeros(self.size)
        idxs = self.uniq_toks[iid]
        if idxs.size == 0:
            return v
        v[idxs] = self.tf_idf[iid]
        return v
import numpy as np
import pandas as pd
from tfidf.tokenizerX import TokenizerX
from numpy.linalg import norm

def calc_tf(doc):
        ret = []
        uniqs = np.unique(doc, return_counts=True)
        #for term in doc:
            #ret.append(doc.count(term)/len(doc))
        for t,c in zip(uniqs[0],uniqs[1]):
            ret.append(c/len(doc))
            #ret.append(doc.count(term)/len(doc))
        return np.array(ret) 

def calc_tf_idf(idf:pd.Series, tf: pd.Series , uniq_toks: pd.Series):
    # iterate rows:
    ret = []
    for i in range(tf.shape[0]):
        toks = uniq_toks.iloc[i]
        tfs = tf.iloc[i]
        idfs = idf.iloc[toks].values
        tf_idf = tfs*idfs
        # normalize
        #tf_idf = tf_idf/norm(tf_idf)
        ret.append(tf_idf)
    return ret


def tokenize_frame(tokenizer:TokenizerX, df:pd.DataFrame, col:str):
    # df_tokens = pd.DataFrame(df.index)#pd.DataFrame(df['item_id'])
    # df_tokens[col+'_tokens'] = df[col].apply(tokenizer.tokenize)
    df_tokens = pd.DataFrame(df[col].apply(tokenizer.tokenize).values, index= df.index, columns=[col+'_tokens'])#pd.DataFrame(df['item_id'])
    
    # stop updating vocab 
    tokenizer.update_mode = False

    # Create vocab dataframe
    df_vocab = pd.DataFrame(tokenizer._word_freq.items()).sort_values(by=1, ascending=False)
    df_vocab.columns = [col+'_word',col+'_freq']
    df_vocab[col+'_word_id'] = df_vocab[col+'_word'].apply(tokenizer.word2idx)
    df_vocab.reset_index(drop=True, inplace=True)
    df_vocab = df_vocab.iloc[:,[2,0,1]]

    #doc freq for vocab
    df_vocab[col+'_doc_freq'] = 0
    df_vocab.sort_values(by=col+'_word_id', inplace=True)
    
    # doc freq for tokens
    for doc in df_tokens[col+'_tokens']:
        doc = set(doc)
        for tok in doc:
            df_vocab.loc[df_vocab[col+'_word_id'] == tok, col+'_doc_freq'] += 1

    # unique tokens
    df_tokens[col+'_uniq_tokens'] = df_tokens[col+'_tokens'].apply(np.unique)
    
    # tf for tokens
    df_tokens[col+'_tf'] = df_tokens[col+'_tokens'].apply(calc_tf)

    

    #idf for vocab
    df_vocab[col+'_idf'] = np.log(len(df_tokens)/df_vocab[col+'_doc_freq'])
    #df_vocab.sort_values(by=col+'_doc_freq', ascending=False, inplace=True)
    # change index to word_id
    df_vocab.set_index(col+'_word_id', inplace=True)


    # tf_idf for tokens
    tf_idf = calc_tf_idf(df_vocab[col+'_idf'], df_tokens[col+'_tf'], df_tokens[col+'_uniq_tokens'])
    df_tokens[col+'_tf_idf'] = tf_idf
    #df_tokens.set_index('item_id', inplace=True)
    df_tokens.sort_index(inplace=True)
    df_vocab.sort_index(inplace=True)
    return df_tokens, df_vocab

def cosine(v1,v2):
    # introduced 0.5 instead of 0 cuz range of cosine for tfidf
    # if sum(v1) == 0 or sum(v2) == 0:
    #     return 0.5
    n = (norm(v1)*norm(v2))
    if n == 0:
        return 0 # 0.5
    return round(np.dot(v1,v2)/n,3)

def cosine_matrix(vs1,vs2 = None):
    # compute cosine similarity for all pairs
    if vs2 is None:
        ret = np.zeros((len(vs1),len(vs1)))
        for i in range(len(vs1)):
            for j in range(len(vs1)):
                if i == j:
                    ret[i,j] = 1
                elif ret[j,i] != 0:
                    ret[i,j] = ret[j,i]
                else:
                    ret[i,j] = cosine(vs1.iloc[i],vs1.iloc[j])    
        ret = pd.DataFrame(ret,index=vs1.index, columns=vs1.index)
    else:
        ret = np.zeros((len(vs1),len(vs2)))
        for i in range(len(vs1)):
            for j in range(len(vs2)):
                ret[i,j] = cosine(vs1.iloc[i],vs2.iloc[j])
        ret = pd.DataFrame(ret,index=vs1.index, columns=vs2.index)
    return ret


def simM_to_multi_indx_df(M:pd.DataFrame):
    h = M.index
    w = M.columns
    ret = pd.DataFrame(data = np.zeros(len(h)*len(w)),
                       index=pd.MultiIndex.from_product([h,w]),
                       columns=['sim'])
    for i,r in M.iterrows():
        ret.loc[i,:] = r.values
    return ret


# def cosine_matrix(vs):
#     # compute cosine similarity for all pairs
#     ret = np.zeros((len(vs),len(vs)))
#     for i in range(len(vs)):
#         for j in range(len(vs)):
#             if i == j:
#                 ret[i,j] = 1
#             elif ret[j,i] != 0:
#                 ret[i,j] = ret[j,i]
#             else:
#                 ret[i,j] = cosine(vs.iloc[i],vs.iloc[j])
    
#     ret = pd.DataFrame(ret,index=vs.index, columns=vs.index)
#     return ret


def normalize(v):
    return v/norm(v)
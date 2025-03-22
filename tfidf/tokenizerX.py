from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
#lemmatization
from nltk.stem import WordNetLemmatizer
import numpy as np

# set for O(1) lookup



class TokenizerX:
    def __init__(self):

        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
        eng_words = set(words.words())
        lemmatizer = WordNetLemmatizer()

        self.stop_words = stop_words
        self.eng_words = eng_words
        self.lemmatizer = lemmatizer

        self._word2idx = {}
        self._idx2word = {}
        self._word_freq = {}
        self.idx = 0
        self.update_mode = True

    def from_vocab(self, wordss, freqs):
        idxs = range(len(wordss))
        self._word2idx = dict(zip(wordss,idxs ))
        self._idx2word = dict(zip(idxs, wordss))
        self._word_freq = dict(zip(wordss, freqs))
        self.idx = len(wordss)
        self.update_mode = False

    def word2idx(self, w):
        if self.update_mode:
            self._update(w)
        return self._word2idx[w]
    
    def idx2word(self, i):
        return self._idx2word[i]
    
    def word2idx_dict(self):
        return self._word2idx
    
    def idx2word_dict(self):
        return self._idx2word

    def tokenize(self, s = "", to_ids = True):
        s = word_tokenize(s)
        ret = []
        for w in s:
            w = w.lower()
            w = self.lemmatizer.lemmatize(w)
            if w in self.stop_words or not w in self.eng_words:
                continue
            if self.update_mode:
                self._update(w)
            if w in self._word2idx:
                ret.append(self._word2idx[w])
            else:
                continue
        # sort ascending
        ret.sort()
        ret = np.array(ret)
        return ret
    
    def _update(self, w):
        if w not in self._word2idx:
            self._word2idx[w] = self.idx
            self._idx2word[self.idx] = w
            self._word_freq[w] = 1
            self.idx += 1
        else:
            self._word_freq[w] += 1

            
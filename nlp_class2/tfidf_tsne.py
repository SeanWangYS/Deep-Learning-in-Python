import json 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

import os
import sys
sys.path.append(os.path.abspath('.'))
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

'''
利用 TF-IDF 技術做出 word vector
並利用 tSNE 做視覺化

Note : 
TF-IDF 當然比不上word2vec 以及 GloVe 做出來的 word vector


Note2 : 
TF-IDF是NLP中用於將文本文檔列表轉換為矩陣表示的常用工具。
每個文檔被轉換為TF-IDF矩陣的row，
並且每個詞存儲在column中。詞彙量（或欄位數）的大小是應該指定的參數，
前5'000-10'000最常見詞的詞彙通常就足夠了。

TF-IDF是稀疏向量，其中向量表示中的非零值的數量總是等於文檔中的唯一單詞的數量。

在擬合期間，tf-idf函數發現語料庫中最常見的單詞並將其保存到詞彙表中。
通過計算詞彙表中每個單詞出現在文檔中的次數來轉換文檔。
因此，tf-idf矩陣將具有[Number_documents，Size_of_vocabulary]的形狀。
每個單詞的權重通過它在語料庫中出現的次數進行歸一化。

'''
### choose a data source ###
# get corpus(Brown) and word2idx
sentences, word2idx = get_sentences_with_word2idx_limit_vocab()
print('finished retriveing data')
print('vocab size:', len(word2idx), 'number of sentences:', len(sentences))

# print(sentences[15]) # 隨便抓一個句子看看 -> [13, 2000, 74, 48, 1578, 26, 19, 2000, 1276, 16, 91, 1143, 160, 2000, 38, 248, 20, 102, 218, 2000, 2000, 27, 15]



## build term document matrix，就像是CountVectorizer()做的事
V = len(word2idx)
N = len(sentences)
A = np.zeros((N, V))   # [Number_documents，Size_of_vocabulary]
print('N:', N, 'V:', V)

for i in range(N):
    for j in sentences[i]:
        A[i, j] += 1
print('finished build term document matrix A, A.shape:', A.shape)

transformer = TfidfTransformer()
A = transformer.fit_transform(A)
print('get TF-IDF matirx, A.shape:', A.shape)
print('type(A)', type(A))   #<class 'scipy.sparse.csr.csr_matrix'>

# tsne requires a dense array
A = A.toarray()  # numpy array

# map back to word in plot
idx2word = { v:k for k, v in word2idx.items()}


## decompose on "documnet axis" by TSNE
tsne = TSNE()
Z = tsne.fit_transform(A.T)
print('get decomposed matrix, Z.shape:', Z.shape)

# get ind2word
idx2word = {v: k for k, v in word2idx.items()}

plt.scatter(Z[:, 0], Z[:, 1])
for i in range(V):
    try:
        plt.annotate(s=idx2word[i].encode('utf8').decode("utf8"), xy=(Z[i, 0], Z[i, 1]))
    except:
        print('bad string:', idx2word[i])
plt.show()
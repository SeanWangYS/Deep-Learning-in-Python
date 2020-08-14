import numpy as np 
from sklearn.metrics.pairwise import pairwise_distances

# WHERE TO GET THE VECTORS:
# GloVe: https://nlp.stanford.edu/projects/glove/
# Direct link: http://nlp.stanford.edu/data/glove.6B.zip


# Euclidean 
def dist1(a, b):
    return np.linalg.norm(a -b)

# cosine
def dist2(a, b):
    return 1 - a.dot(b) / np.linalg.norm(a) * np.linalg.norm(b)

# pick a  distance type
dist, metric = dist2, 'cosine'

## more intuitive
# def find_analgise(w1, w2, w3):
#     for w in (w1, w2, w3):
#         if w not in word2vec:
#             print("%s not in dictionary" % w)
#             return
    
#     king = word2vec[w1]
#     man = word2vec[w2]
#     woman = word2vec[w3]
#     v0 = king - man + woman

#     min_dist = float('inf')
#     best_word = ''
#     for word, v1 in word2vec.tiems():
#         if word not in (w1, w2, w3):
#             d = dist(v0, v1)
#             if d < min_dist:
#                 min_dist = d
#                 best_word = word
#     print(w1, '-', w2. '=', best_word, '-', w3)

## fast
# 針對一組字，找最佳類比單字
def find_analogies(w1, w2, w3):
    for w in (w1, w2, w3):
        if w not in word2vec:
            print("%s not in dictionary" % w)
            return

    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman

    distances = pairwise_distances(v0.reshape(1, D), embedding, metric=metric).reshape(V)
    print(type(distances))   # <class 'numpy.ndarray'>
    print(len(distances))    # 400000
    idxs = distances.argsort()[:4]    # argsort函数返回的是数组值从小到大的索引值
    for idx in idxs:
        word = idx2word[idx]
        if word not in (w1, w2, w3):
            best_word = word
            break
    
    print(w1, "-", w2, "=", best_word, "-", w3)

# 查詢單一單字，找n個最鄰近字(期望有相同辭意)
def nearest_neighbors(w, n=5):
    if w not in word2vec:
        print("%s not in dictionary" % w)
        return
    
    v = word2vec[w]
    distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[1:n+1]       # 為什麼從1開始抓，因為0 應該會是查詢的單字本身
    print('neighbors of %s' % w)
    for idx in idxs:
        print("\t%s" % idx2word[idx])



# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
embedding = []    # word embedding matrix
idx2word = []
with open('large_files/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.array(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))
    embedding = np.array(embedding)
    V, D = embedding.shape



find_analogies('king', 'man', 'woman')
# find_analogies('france', 'paris', 'london')
# find_analogies('france', 'paris', 'rome')
# find_analogies('paris', 'france', 'italy')
# find_analogies('france', 'french', 'english')
# find_analogies('japan', 'japanese', 'chinese')
# find_analogies('japan', 'japanese', 'italian')
# find_analogies('japan', 'japanese', 'australian')
# find_analogies('december', 'november', 'june')
# find_analogies('miami', 'florida', 'texas')
# find_analogies('einstein', 'scientist', 'painter')
# find_analogies('china', 'rice', 'bread')
# find_analogies('man', 'woman', 'she')
# find_analogies('man', 'woman', 'aunt')
# find_analogies('man', 'woman', 'sister')
# find_analogies('man', 'woman', 'wife')
# find_analogies('man', 'woman', 'actress')
# find_analogies('man', 'woman', 'mother')
# find_analogies('heir', 'heiress', 'princess')
# find_analogies('nephew', 'niece', 'aunt')
# find_analogies('france', 'paris', 'tokyo')
# find_analogies('france', 'paris', 'beijing')
# find_analogies('february', 'january', 'november')
# find_analogies('france', 'paris', 'rome')
# find_analogies('paris', 'france', 'italy')

# nearest_neighbors('king')
# nearest_neighbors('france')
# nearest_neighbors('japan')
# nearest_neighbors('einstein')
# nearest_neighbors('woman')
# nearest_neighbors('nephew')
# nearest_neighbors('february')
# nearest_neighbors('rome')

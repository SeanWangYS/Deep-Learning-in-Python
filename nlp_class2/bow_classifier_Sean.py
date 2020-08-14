import numpy as np 
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

'''
這一篇的目的，是為每一個文本做出對應的feature vector ，搭配每個文本的分類Label
就可以拿來訓練模型

重點在於，
把文本內每個詞彙的詞向量，做疊加並取平均來創造美個文本的feature vector

data base
https://www.cs.umb.edu/~smimarog/textmining/datasets/
'''

## load pre-train embedding matrix =========================================
print('Loading word vectors...')
word2vec = {}
embedding = []
idx2word= []
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
    # print('Found %s word vectors.' % len(word2vec))
    embedding = np.array(embedding)
    V, D = embedding.shape


## transfer training material into features and label===============================
def get_data(path='large_files/r8-train-all-terms.txt'):
    labels = []
    features = []
    with open(path, encoding='utf8') as f:
        for line in f:
            values = line.split("\t")
            
            # collect label
            labels.append(values[0])

            # collect feature
            text  = values[1]
            words = text.split()
            words_len = len(words)  
            vec = np.zeros(50, dtype='float')
            for word in words:
                # print(type(word), word)
                if word in word2vec:
                    # print(word2vec[word])
                    vec += np.array(word2vec[word])
                else: 
                    words_len -=1
            vec_nor = vec / words_len
            features.append(vec_nor)

    list_of_label = list(set(labels))
    # print('list_of_label', list_of_label)
    # label2idx = {list_of_label[i]: i for i in range(len(list_of_label))}
    label2idx = {k:v for v,k in enumerate(list_of_label) }  # 更聰明的寫法
    # print(label2idx.items())
    labels = [label2idx[label] for label in labels]
    

    # print('check training set and test set==========================================')
    # print(len(features))
    # print(features[0])
    # print(len(labels))
    # print(labels[0])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

print('loading traning data and test data...')
Xtrain ,Ytrain = get_data('large_files/r8-train-all-terms.txt')
Xtest, Ytest = get_data('large_files/r8-test-all-terms.txt')
print('Xtrain.shape:', Xtrain.shape)
print('Ytrain.shape:', Ytrain.shape)


model = RandomForestClassifier(n_estimators=200)     # test score: 0.9355870260392873
# model = XGBClassifier()     # test score: 0.8072179077204202
model.fit(Xtrain, Ytrain)
print('train score:', model.score(Xtrain, Ytrain))
print('test score:', model.score(Xtest, Ytest))




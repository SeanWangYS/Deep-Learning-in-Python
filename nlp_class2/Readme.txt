程式撰寫順序如下

========================= word-vector and word analogy ================================
theory and concept:
* word vector 
* word analogy

tfidf_tsne.py
    利用 TF-IDF 技術做出 word vector，並利用 tSNE 做視覺化
pretrained_glove.py
    用套件載入 glove wordvect，做word analogy 比較
pretrained_w2v.py
    用套件載入 word2vec，做word analogy 比較
bow_classifier.py
    這一篇的目的，是為每一個文本做出對應的feature vector ，搭配每個文本的分類Label，就可以拿來訓練分類模型
    重點在於，把文本內每個詞彙的詞向量，做疊加並取平均來創造美個文本的feature vector


================================ language modeling ===================================
theory and concept:
* language model
    基本上是 p(current word | previous word) 模型
    可以把 hidden layer 當作 wordvector
    也就是說訓練一個language model 的目的是，取得一組好用的 word2vec
* Biagram model


 markov.py
    用markov 概念，建立一個 bigram model (最簡單的語言模型)
logistic_Sean.py
    Neural Bigram Model in Code，用logistic 架構，
    也就是最基本的neural unit，來 train 一個 Bigrame model
neural_network_Sean.py
    Neural Network Bigram Model
    用 NN 架構(一層hidden layer)
    用到auto-encoder的觀念，來 train 一個 Bigrame model
    (對這支code有做實驗，紀錄於code中，值得看)
neural_network2_Sean.py
    承接上一支code
    這一篇加上了 numpy 中 indexing trick 的技巧，可以加速計算

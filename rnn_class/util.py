import numpy as np 
import string 
import os 
import sys 
import operator 
from nltk import pos_tag, word_tokenize 
from datetime import datetime

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

def all_parity_pairs(nbit):
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember. 
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder   # Ntotal要刻意湊到100的倍數
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y

def all_parity_pairs_with_sequence_labels(nbit):
    X, Y = all_parity_pairs(nbit)
    N, t = X.shape

    # we want every time step to have a label
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)
    return X, Y_t

def remove_punctuation_3(s):  # for python3的寫法  # input 參數，要是str
    return s.translate(str.maketrans('','',string.punctuation))  # 把所有標點符號用None取代掉

if sys.version.startswith('3'):
    remove_punctuation = remove_punctuation_3     # 問育叡，從48行到53行，這種程式的寫法，不是很懂

## 取得詩文本poem，包含word2idx 以及 轉成索引指標後的句子 的 list
def get_robert_frost(): 
    word2idx = {'START':0, 'END':1}   # define "strat token" and "end token"，give them the magic numbers we like
    current_idx = 2                   # 目前word2idx 中 出現的index 有0, 1 ，下一個index 是2
    sentences = []
    with open('.\\rnn_class\\robert_frost.txt', 'r', encoding='utf-8') as f:    # 注意，這邊用二進位制讀取'rb' (讀取成byte型別)，或著加上encoding='utf-8'(讀取成str)
        for line in f:
            line = line.strip()
            if line:
                tokens = remove_punctuation(line.lower()).split()
                sentence = []        # sentence 裡面是每個句子中的每個字，轉成idx 後的有序排列
                for t in tokens:
                    if t not in word2idx:
                        word2idx[t] = current_idx
                        current_idx += 1
                    idx = word2idx[t]
                    sentence.append(idx)
                sentences.append(sentence)
    return sentences, word2idx

def get_tags(s):
    tuples = pos_tag(word_tokenize(s))  # 單一tuple=> ('true', 'JJ')，第二個元素就是pos_tag
    return [y for x, y in tuples] # 我們只取 pos_tag

'''
note for get_poetry_classifier_data
samples_per_class 的目的 => you might want to limit this when you're testing your code because we're tokenization takes very long.
'''
def get_poetry_classifier_data(samples_per_class, load_cached=True, save_cached=True):
    datafile = './rnn_class/poetry_classifier_data.npz'
    if load_cached and os.path.exists(datafile):
        npz = np.load(datafile, allow_pickle=True)
        X = npz['arr_0']
        Y = npz['arr_1']
        V = int(npz['arr_2'])
        return X, Y, V
    
    word2idx = {} # actually, it's a pos index
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('./rnn_class/edgar_allan_poe.txt', './rnn_class/robert_frost.txt'), (0, 1)):
        count = 0
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()  # rstrip() 只移除str右邊的空白 ; strip() 移除str左右兩邊的空白
                if line:
                    print(line)
                    # tokens = remove_punctuation(line.lower()).split()
                    tokens = get_tags(line)      # 這裡的tokens 是由原來句子中每個單字的 pos-tags 組成的一個list
                    if len(tokens) > 1:
                        # scan doesn't work nice here, technically could fix...
                        for token in tokens:
                            if token not in word2idx:
                                word2idx[token] = current_idx
                                current_idx += 1
                        sequence = np.array([word2idx[w] for w in tokens])
                        X.append(sequence)
                        Y.append(label)
                        count += 1
                        print(count)
                        if count >= samples_per_class:
                            break
    if save_cached:
        np.savez(datafile, X, Y, current_idx)
    return X, Y, current_idx


def get_wikipedia_data(n_files, n_vocab, by_paragraph=False):
    prefix = '../large_files/'

    if not os.path.exists(prefix):
        print("Are you sure you've downloaded, converted, and placed the Wikipedia data into the proper folder?")
        print("I'm looking for a folder called large_files, adjacent to the class folder, but it does not exist.")
        print("Please download the data from https://dumps.wikimedia.org/")
        print("Quitting...")
        exit()

    input_files = [f for f in os.listdir(prefix) if f.startswith('enwiki') and f.endswith('txt')]

    if len(input_files) == 0:
        print("Looks like you don't have any data files, or they're in the wrong location.")
        print("Please download the data from https://dumps.wikimedia.org/")
        print("Quitting...")
        exit()

    # return variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}

    if n_files is not None:
        input_files = input_files[:n_files]

    for f in input_files:
        print("reading:", f)
        for line in open(prefix + f):
            line = line.strip()
            # don't count headers, structured data, lists, etc...
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                if by_paragraph:
                    sentence_lines = [line]
                else:
                    sentence_lines = line.split('. ')
                for sentence in sentence_lines:
                    tokens = my_tokenizer(sentence)
                    for t in tokens:
                        if t not in word2idx:
                            word2idx[t] = current_idx
                            idx2word.append(t)
                            current_idx += 1
                        idx = word2idx[t]
                        word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
                    sentence_by_idx = [word2idx[t] for t in tokens]
                    sentences.append(sentence_by_idx)

    # restrict vocab size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    assert('START' in word2idx_small)
    assert('END' in word2idx_small)
    assert('king' in word2idx_small)
    assert('queen' in word2idx_small)
    assert('man' in word2idx_small)
    assert('woman' in word2idx_small)

    # map old idx to new idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small

if __name__ == "__main__":
    # X, Y = all_parity_pairs(12)
    # print('X: \n', X)
    # print('Y: \n', Y)


    X, Y = all_parity_pairs_with_sequence_labels(4)
    print("X.shape: ",X.shape)
    print('Y.shape: ',Y.shape)
    print('X: \n', X)
    print('Y: \n', Y)

    # print(os.getcwd())

    # sentences, word2idx = get_robert_frost()
    # print('len(sentences):',len(sentences))
    # print( 'len(word2idx):', len(word2idx))
    # print(sentences[:3])

    # X, _, _ = get_poetry_classifier_data(10, True, True)     # 出來的資料結構
    # print(X)

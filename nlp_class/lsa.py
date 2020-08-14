import nltk
import numpy as np 
import matplotlib.pyplot as plt 

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer() 


'''
這一篇的目的: 初探 LSA (latten samentic analysis)
最原始的方法做出一篇corpus 的字向量  也就是 word-frequency vector

材料是許多書的title，title內有許多token(字符)
step1. 將每個title 經過前處裡
    把每一個字轉字根
    remove stopwords...等等

step2. 製作 word-to-index map 以及 index-to-word map

step3. 製作 word-frequency vectors

step4. 用SVD降維並且畫圖看有沒有同義字群聚現象，
        注意:這裡word-frequency vectors 組成的 matrix在 SVD降維前  沒有做標準化 
'''

titles = [line.rstrip() for line in open("./nlp_class/all_book_titles.txt")]

stopwords = set(w.rstrip() for w in open("./nlp_class/stopwords.txt"))

# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# add more stopwords specific to this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth',})

def my_tokenizer(s):
    s = s.lower()   # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form (把每一個字轉成字根的型態)
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]  # remove any digits, i.e. "3rd edition"
    return tokens    # 每個tokens，都是一個title的拆解，拆解後回傳list


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
error_count = 0
for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)      # all_tokens內單一元素是list，代表文章title，所以會有重複的token存入，all_tokens就是拿來做word vector的原材料
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        error_count += 1

print("number of errors parsing file:", error_count, "number of lines in file:", len(titles))
if error_count == len(titles):
    print('There is no data to do anything with! Quitting...')
    exit()

# now let's create our input matrices - just indicator variables for this example - works better than proportions
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        x[word_index_map[t]] +=1
    return x
    
N = len(all_tokens)  # 所有書書的書名數量 = N
D = len(word_index_map)
X = np.zeros((D, N))   # terms will go along rows, documents along columns
i = 0
for tokens in all_tokens:
    X[:, i] = tokens_to_vector(tokens)

def main():
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
    plt.show()


if __name__ == '__main__':
    main()
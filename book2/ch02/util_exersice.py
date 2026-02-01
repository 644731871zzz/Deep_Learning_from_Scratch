import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import os
from common.np import *

def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus,word_to_id,id_to_word


def create_co_matrix(corpus,vocab_size,window_size = 1):#
    """生成共共现矩阵
    corpus:语料库(id形式)
    vocab_size:词汇个数  一般提前算好,len(word_to_id)
    window_size:窗口大小
    """ #有四种字符串写法 'a' "a" '''a''' """a"""都是等价的,后者两个可以使用三个引号的'和"可以换行
    corpus_size = len(corpus)#语料库总单词个数
    co_matrix = np.zeros((vocab_size,vocab_size),dtype = np.int32)#初始化共现矩阵 int32指的是每个元素(每个格子)

    for idx,word_id in enumerate(corpus):#遍历每一个单词
        for i in range(1,window_size + 1):#
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >=0:#确保在句子中
                left_word_id = corpus[left_idx]#记录id
                co_matrix[word_id,left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] += 1

    return co_matrix


def cos_similarity(x,y,eps = 1e-8):
    '''
    cos_similarity 的 Docstring
    
    :param x: 向量
    :param y: 向量
    :param eps: 放置除数为0的微小值 epsilon表示非常小的正数
    '''
    nx = x / (np.sqrt(np.sum(x**2)) + eps)#正规化
    ny = y / (np.sqrt(np.sum(y**2)) + eps)

    return np.dot(nx,ny)


def most_similar(query,word_to_id,id_to_word,word_matrix,top = 5):
    """相似单词查找"""

    if query not in word_to_id:
        print('%s is not found' % query)
        return#结束函数立即返回 空return(当前)和没写都表示返回None
    
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i],query_vec)

    count = 0
    for i in (-1*similarity).argsort():#先求反,然后用升序返回索引列表(非惰性)   numpy没有内置降序取反求升序是默认做法
        if id_to_word[i] == query:
            continue#跳过当前这次循环,进入下次for继续执行(后面的语句不会执行)
        print(' %s: %s'% (id_to_word[i],similarity[i]))

        count +=1#累加只用 += 没有c的歧义
        if count >= top:
            return
        

def ppmi(C,verbose = False,eps = 1e-8):
    M = np.zeros_like(C,dtype = np.float32)
import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *


def analogy(a,b,c,word_to_id,id_to_word,word_matrix,top = 5,answer = None):
    for word in (a,b,c):
        if word not in word_to_id:
            print('%s is not in found' %word)
            return
        
    print('\n[analogy] ' + a + ':' + b + '=' + c + ':?')
    a_vec,b_vec,c_vec = word_matrix[word_to_id[a]],word_matrix[word_to_id[b]],word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    #如果长度为0,可能产生nan值
    similarity = np.dot(word_matrix,query_vec)

    if answer is not None: #answer如果输入为标准答案,输出相似度
        print('===>' + answer + ':' + str(np.dot(word_matrix[word_to_id[answer]],query_vec)))

    #记录打印了几个结果
    count = 0
    #argsort()是升序, *-1这里取了反,运行argsort后所以最相似排在上面,输出了对应索引的列表
    for i in (-1 * similarity).argsort():
        #跳过nan值
        if np.isnan(similarity[i]): 
            continue
        if id_to_word[i] in (a,b,c):
            continue
        print(' {0} : {1}'.format(id_to_word[i],similarity[i]))

        count += 1
        if count >= top:
            return
        






#归一化,只关注方向
def normalize(x):
    if x.ndim == 2:
        #sum(1)等价于sum(axis = 1)
        #axis = 1表示压缩axis = 1的维度,这里是行,压缩行就是行方向相加
        s = np.sqrt((x*x).sum(1))
        x /= s.reshape((s.shape[0],1))
    elif x.ndim == 1:
        s = np.sqrt((x*x).sum())
        x /= s
    return x
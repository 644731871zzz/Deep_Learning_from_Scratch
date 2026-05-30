import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from common.util import preprocess,create_co_matrix,cos_similarity,ppmi 

text = 'You say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)

#配置numpy打印三位小数,使用print打印numpy会自动替换,print本质是先将其转化成字符串,这时候调用numpy自己的方法
np.set_printoptions(precision = 3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)
import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *


def create_contexts_target(corpus,window_size = 1):
    #切片不包含终点,到-window_size之前一个结束,想取到最后一个直接[a:]不是[a:-1]
    target = corpus[window_size: - window_size] 
    contexts = []

    #len(corpus) - 1才是最后一个单词
    for idx in range(window_size,len(corpus) - window_size):
        cs = []
        for t in range(-window_size,window_size + 1):
            if t == 0:
                continue #跳过当前循环进入下一次循环
            cs.append(corpus[idx + t])
        contexts.append(cs) #

    return np.array(contexts),np.array(target)
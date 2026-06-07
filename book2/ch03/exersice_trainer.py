import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy
import time
import matplotlib.pyplot as plt
from common.np import *
from common.util import clip_grads

def remove_duplicate(params,grads):
    params,grads = params[:],grads[:]

    #循环,一次循环检查一个重复项,直到没有
    while True:
        find_flg = False
        L = len(params) #获取有多少组params

        for i in range(0,L - 1):
            for j in range(i+1,L):
                #is判断对象是不是同一个对象,不是判断是否相等,是不是引用
                if params[i] is params[j]:
                    #重复的梯度合并
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j) #pop()是删除这个对应元素,并返回这个元素,这里没接受返回
                    grads.pop(j)

                elif params[i].ndim ==2 and params[j].ndim ==2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]): #
                    grads[i] += grads[j].T
                    find_flg = True
                    #只是修改索引数值的外部维度怎么做,不是修改内部数值(内部数值修改全都变了)
                    params.pop(j) 
                    grads.pop(j)

                if find_flg:break #break结束整个循环,continue是结束当前循环
            if find_flg:break

        if not find_flg:break

    return params,grads
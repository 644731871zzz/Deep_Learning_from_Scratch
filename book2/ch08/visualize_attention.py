import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from dataset import sequence
import matplotlib.pyplot as plt
from attention_seq2seq import AttentionSeq2seq

(x_train,t_train),(x_test,t_test) = \
    sequence.load_data('date.txt') #\后不强制缩进,为了可读性需要缩进
char_to_id,id_to_char = sequence.get_vocab()

x_train,x_test = x_train[:,::-1],x_test[:,::-1]

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256

model = AttentionSeq2seq(vocab_size,wordvec_size,hidden_size)
model.load_params()

_idx = 0
def visualize(attention_map,row_labels,columns_labels):
    fig,ax = plt.subplots()
    #pccolor绘制热图,比sns的heatmap更原始
    ax.pcolor(attention_map,cmap = plt.cm.Greys_r,vmin = 0.0,vmax = 1.0)

    ax.patch.set_facecolor('black') #背景配置为黑色,patch理解为坐标轴后面的那'块'背景
    ax.set_yticks(np.arange(attention_map.shape[0]) + 0.5, minor = False)
    ax.set_xticks(np.arange(attention_map.shape[1]) + 0.5, minor = False)
    ax.invert_yaxis()#控制坐标翻转,为了让0在左上角
    ax.set_xticklabels(row_labels,minor = False)
    ax.set_yticklabels(columns_labels,minor = False)

    global _idx #global调用最外层变量,拿进内部变量,变量和变量名不变
    _idx += 1 #每次画图,_idx + 1
    plt.show()


np.random.seed(1984)
for _ in range(5):
    idx = [np.random.randint(0,len(x_test))]
    x = x_test[idx]
    t = t_test[idx]
    model.forward(x,t)
    #a的形状是(N,T_enc),N是因为有N簇,这里是N = 1
    #d是3d,因为a是2d数据,叠加了多个a
    d = model.decoder.attention.attention_weights
    d = np.array(d)
    #a的N = 1,所以可以reshape这个形状
    attention_map = d.reshape(d.shape[0],d.shape[2])
    #将之前的翻转正过来
    attention_map = attention_map[:,::-1]
    x = x[:,::-1]

    row_labels = [id_to_char[i] for i in x[0]]
    column_labels = [id_to_char[i] for i in t[0]]
    column_labels = column_labels[1:]

    visualize(attention_map,row_labels,column_labels)
import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset import spiral
import matplotlib.pyplot as plt

x,t = spiral.load_data()
print('x',x.shape)
print('t',t.shape)#t是one-hot向量,正确标签为一在对应索引上,其余标签为0

N = 100
CLS_NUM = 3
markers = ['o','x','^']#最后是三角形
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N,0],x[i*N:(i+1)*N,1],s = 40,marker = markers[i])

plt.show()
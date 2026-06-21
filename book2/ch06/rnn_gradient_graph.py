import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

N = 2
H = 3
T = 20

dh = np.ones((N,H))

np.random.seed(3)

#Wh = np.random.randn(H,H)
Wh = np.random.randn(H,H) * 0.5


norm_list = []
for t in range(T):
    dh = np.dot(dh,Wh.T)
    #这是用大矩阵当成一个向量,近似当成L2范数,结果的描述可以近似,但是不严谨
    norm = np.sqrt(np.sum(dh ** 2)) / N
    #norm = np.mean(np.sqrt(np.sum(dh ** 2, axis=1)))
    norm_list.append(norm)

print(norm_list)

plt.plot(np.arange(len(norm_list)),norm_list)
plt.xticks([0,4,9,14,19],[1,5,10,15,20])
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()
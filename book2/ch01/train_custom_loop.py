import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x,t = spiral.load_data()
model = TwoLayerNet(input_size=2,hidden_size = hidden_size,output_size=3)
optimizer = SGD(lr = learning_rate)

data_size = len(x)
max_iters = data_size//batch_size#//是整除符号,不要小数点后面的部分
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)#生成从0到data_size - 1的数的随机排列,不重复,长度为data_size
    #同时打乱x,t的顺序,新的x和t的对应关系依旧保持
    x = x[idx]
    t = t[idx]
    
    for iters in range(max_iters):
        batch_x = x[iters * batch_size:(iters + 1)*batch_size]
        batch_t = t[iters * batch_size:(iters + 1)*batch_size]

        #最后一层计算softmax+交叉熵误差后计算出的,下界是0,上界不封顶,所以损失函数会有大于一的情况
        loss = model.forward(batch_x,batch_t)
        model.backward()
        optimizer.update(model.params,model.grads)

        total_loss += loss
        loss_count += 1#

        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f'
                  %(epoch + 1,iters + 1,max_iters,avg_loss))#注意这是显示平均损失,不是准确率,一会绘图也是,所以有大于1的情况
            loss_list.append(avg_loss)
            total_loss,loss_count = 0,0


#绘制学习结果
plt.plot(np.arange(len(loss_list)),loss_list,label = 'train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()#结束当前fig并显示,所以后面的plt将会默认创建新图

#绘制决策边界
h = 0.001
x_min,x_max = x[:,0].min() - .1,x[:,0].max() + .1
y_min,y_max = x[:,1].min() - .1,x[:,1].max() + .1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))#h是step = h
X = np.c_[xx.ravel(),yy.ravel()]#X形状为(N,2),列方向
score = model.predict(X)
predict_cls = np.argmax(score,axis = 1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx,yy,Z)
plt.axis('off')

#绘制数据点
x,t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o','x','^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N,0],x[i*N:(i+1)*N,1],s = 40,marker = markers[i])
plt.show()
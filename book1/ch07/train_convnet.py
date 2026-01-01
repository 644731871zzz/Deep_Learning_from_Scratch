import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer #导入训练循环,簇,更新参数,记录数据等封装起来的类

(x_train,t_train),(x_test,t_test) = load_mnist(flatten = False)

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param= {'filter_num' : 30,'filter_size':5,'pad':0,'stride':1},#滤波器个数30个
                        hidden_size=100,output_size=10,weight_init_std=0.01)

trainer = Trainer(network,x_train,t_train,x_test,t_test,
                  epochs=max_epochs,mini_batch_size=100,
                  optimizer='Adam',optimizer_param={'lr':0.001},#lr表示学习率,这里的值是adam的默认值
                  evaluate_sample_num_per_epoch=1000)#简单计算成功率,用前1000个样本计算成功率

trainer.train()
network.save_params('params.pkl')
print('Saved Network Parameters!')

markers = {'train':'o','test':'s'}
x = np.arange(max_epochs)
plt.plot(x,trainer.train_acc_list,marker = 'o',label = 'train',markevery = 2)
plt.plot(x,trainer.test_acc_list,marker = 's',label = 'test',markevery = 2)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0,1.0)
plt.legend(loc = 'lower right')
plt.show()
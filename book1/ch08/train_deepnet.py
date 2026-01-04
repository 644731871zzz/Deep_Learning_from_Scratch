import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train,t_train),(x_test,t_test) = load_mnist(flatten = False)

network = DeepConvNet()
trainer = Trainer(network,x_train,t_train,x_test,t_test,
                  epochs = 20,mini_batch_size = 100,
                  optimizer= 'Adam',optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)#只取1000个样本计算正确率

trainer.train()

network.save_params('deep_convnet_params.pkl')
print('Save Network Parameters!')
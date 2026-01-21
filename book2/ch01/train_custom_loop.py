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
max_iters = data_size//batch_size#
total_loss = 0#
loss_count = 0#
loss_list = []


import numpy as np
import matplotlib.pyplot as plt
from gradient_2d import numerical_gradient


def gradient_descent(f,init_x,lr = 0.01,step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f,x)
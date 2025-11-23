import numpy as np

def shuffle_dataset(x,t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]#else这个是硬写的,没必要.这个也只能为4d工作不能为其他维度工作
    t = t[permutation]

    return x,t
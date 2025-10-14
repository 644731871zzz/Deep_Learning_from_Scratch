import numpy as np

def _numerical_gradient_1d(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    #size表示得知所有元素的总数,因为最后在内存中都是线性排列,仅仅读取元素个数.(非外层维度个数)
        #这里是一维数组与上者无关
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val

    return grad

def numerical_gradient_2d(f,X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f,X)
    else:
        grad = np.zeros_like(X)

        #enumerate遍历可迭代对象,返回索引值和当前元素
            #这里返回的当前元素仅仅为最外层维度的遍历,不是最内层1d上的元素
            #按照第一维度(axis = 0)(不是1d)的切片遍历数组
        for idx,x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f,x)

        return grad

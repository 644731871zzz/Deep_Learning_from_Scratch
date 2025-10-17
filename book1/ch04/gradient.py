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
    
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    #一个多维数组迭代器,依次访问数组中每个元素 返回一个nditer迭代器对象
    #flags指的是将多维下标(对应索引)存在字符串名称中  op_flags表示允许迭代过程中修改值(默认是只读)
    #参数赋值的[]是配置项,不是属性赋值  为[]是因为参数可以接收很多个选项名
    it = np.nditer(x,flags = ['multi_index'],op_flags=['readwrite'])
    #直到遍历完之前一直循环
    while not it.finished:
        #得到当前元素多维索引,返回索引元组
            #numpy支持用元组用多维索引 [()]索引等价于[]索引
        idx = it.multi_index
        #提取处理元素
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        #移动到下一个元素
        it.iternext()

    return grad


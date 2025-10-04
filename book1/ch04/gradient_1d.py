import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f,x):
    #计算斜率
    d = numerical_diff(f,x)
    print(d)
    #利用斜率求对应点的斜率的直线方程. y = kx + b 这个b对应下面的y
    y = f(x) - d*x
    #匿名函数语法 lambda 参数:表达式  构建一个函数 输入值为t 返回值为:后的语句
    #此时没有执行,没有函数名后面被赋值给tf时候tf才是函数名(使用lambda构建的开始)
    return lambda t: d*t + y

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')

tf = tangent_line(function_1,5)
y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()
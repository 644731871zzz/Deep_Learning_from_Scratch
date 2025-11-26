import numpy as np

def im2col(input_data,filter_h,filter_w,stride = 1,pad = 0):

    """
    im2col 的 Docstring
    
    :param input_data: (数据量,通道,高,长)的4d数组
    :param filter_h: 滤波器的高
    :param filter_w: 滤波器的长
    :param stride: 步幅
    :param pad: 填充

    返回2d数组
    """

    N,C,H,W = input_data.shape
    #公式直接求出来通道中的一层输出大小,也是一个滤波器对一个图做完卷积运算的输出大小
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    #pad函数给外围加0
    #函数第一个表示添加0的数据  constant表示用0填充,参数形参为mode  填充其他数需要用constant_values = ?的参数
    # 第二个参数表示每个维度添加多少个单位单位  列表中单位从外到内
        #(,)一个维度只有两个端点,左边表示起点右边表示重点,按照索引来
        #填充单位需要根据当前轴下的元素的形状为准
    #(0,0)表示前两个维度不补零
    #(pad,pad)表示填充pad个数量
    img = np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    #初始化4d操作后的输出值,这是个暂存窗口,是仓库的作用
    col = np.zeros((N,C,filter_h,filter_w,out_h,out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h#
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:,:,y,x,:,:] = img[:,:,y:y_max:stride,x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col
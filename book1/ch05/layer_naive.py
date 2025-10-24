class MulLayer:
    def __init__(self):
        #反向传播需要正向传播的值
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y

        return out
    
    def backward(self,dout):
        dx = dout*self.y
        dy = dout*self.x

        return dx,dy
    

class AddLayer:
    def __init__(self):
        pass #什么也不执行

    def forward(self,x,y):
        out = x+y
        
        return out
    
    def backwark(self,dout):

        dx = dout*1 #数学说明写法,表示强调不改变值
        dy = dout*1

        return dx,dy
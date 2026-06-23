import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *

def eval_perplexity(model,corpus,batch_size = 10,time_size = 35):
    print('evaluation perplexity...')
    corpus_size = len(corpus)
    #total_loss是累计loss的意思,这里进行初始化
    total_loss,loss_cnt = 0,0
    #计算迭代多少次
    #//是除法,向下取整
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    #可以理解为每一簇多少个,控制每个batch的起始间隔位置
    jump = (corpus_size - 1) // batch_size 

    for iters in range(max_iters):
        #初始化数据和监督标签的输入
        xs = np.zeros((batch_size,time_size),dtype = np.int32)
        ts = np.zeros((batch_size,time_size),dtype = np.int32)
        #起始位置向后移动
        time_offset = iters * time_size
        #列出每个batch的起始位置
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        #输出当前时间块里面的所有xs和ts
        for t in range(time_size):
            for i,offset in enumerate(offsets):
                #%corpus_size防止越界,如果越界就索引开头
                xs[i,t] = corpus[(offset + t) % corpus_size]
                ts[i,t] = corpus[(offset + t + 1) % corpus_size] 


        try:
            #train_flg表示dropout开关
            loss = model.forward(xs,ts,train_flg = False)
        except TypeError:
            #如果模型不支持dropout,用普通的forward
            loss = model.forward(xs,ts)
        #累加当前的loss
        total_loss += loss

        #打印进度
        #输出在命令行,\r表示光标回到开头,而不是换行(print也可以用)
        sys.stdout.write('\r %d / %d' %(iters,max_iters))
        #将缓冲区的内容输出
        sys.stdout.flush()

    print('')#换行
    ppl = np.exp(total_loss / max_iters)#输出困惑度
    return ppl
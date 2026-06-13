import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common import config
from common.np import *
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target,to_cpu,to_gpu
from dataset import ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus,word_to_id,id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts,target = create_contexts_target(corpus,window_size)

if config.GPU:
    contexts,target = to_gpu(contexts),to_cpu(target)

model = CBOW(vocab_size,hidden_size,window_size,corpus)
optimizer = Adam()
trainer = Trainer(model,optimizer)

trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()

#在negative_sampling_layer里面在训练时修改了定位到的最底部数据,这里直接读取即可
word_vecs = model.word_vecs 
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
#映射的字典也保存了
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
#打开一个pkl文件,wb是write模式,写入二进制文件
    #open是python的语法,可以创建一个可写入文件,可以打卡读取文件
    #这个文件不是容器,但是with支持,并支持结束关闭
#as f将这个文件指向对象名f
#wb模式一定是写入,如果这个文件里面有内容,将会删除在写入(覆盖)
#根据第二个字符串决定是什么形式,open的意思是定位到读写哪个文件
    # 'rb'  读取二进制，文件必须存在
    # 'wb'  写入二进制，不存在就创建，存在就清空
    # 'ab'  追加二进制，不存在就创建
    # 'xb'  新建二进制，文件已存在就报错
with open(pkl_file,'wb') as f:
    #将params以二进制格式存入f -1表示最高版本的pickle协议
    pickle.dump(params,f,-1)
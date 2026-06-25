import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset import sequence

#\是换行继续写的意思,而且\是这一行最后的有效字符.\的下一行语句必须是和这一行的延伸
(x_train,t_train),(x_test,t_test) = \
    sequence.load_data('addition.txt',seed = 1984) 

#character字符的意思
char_to_id,id_to_char = sequence.get_vocab()

print(x_train.shape,t_train.shape)
print(x_test.shape,t_test.shape)

print(x_train[0])
print(t_train[0])

print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))
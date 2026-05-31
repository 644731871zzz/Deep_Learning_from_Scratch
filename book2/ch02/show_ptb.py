import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset import ptb

corpus,word_to_id,id_to_word = ptb.load_data('train')#导入训练集

print('corpus size',len(corpus))
print('corpus[:30]',corpus[:30])
print()
print('id_to_word[0]:',id_to_word[0])
print('id_to_word[1]:',id_to_word[1])
print('id_to_word[2]:',id_to_word[2])
print()
print('word_to_id["car"]:',word_to_id['car'])
print('word_to_id["happy"]:',word_to_id['happy'])
print('word_to_id["lexus"]:',word_to_id['lexus'])
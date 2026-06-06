# coding: utf-8
import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi
from common.util import create_contexts_target


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)


context,target = create_contexts_target(corpus)

print(context)
print(target)

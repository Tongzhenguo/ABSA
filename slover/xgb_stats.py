'''
基于统计特征的分类模型
'''
import os

import jieba
import pandas as pd

aspect_path = '../cache/aspect_list.pkl'
pos_sem_path = '../cache/pos_semtiment_words_list.pkl'
neg_sem_path = '../cache/neg_semtiment_words_list.pkl'

data = pd.read_excel('../data/BDCI2017-+泰一指尚+-测试数据（非完整数据）.xlsx',sheetname='Sheet5')
words = set()
def add_words(s):
    for w in list(jieba.cut(str(s), HMM=False)):
        if w not in words:
            words.add(w)

'''
    存在有的词是主题有的时候又不是的情况
'''
def make_train_data():
    path = '../cache/xgb_train_stats.pkl'
    if os.path.exists( path ):
        return pd.read_pickle( path )
    else:
        data['content-评论内容'].apply( add_words )
        aspect_list = pd.read_pickle(aspect_path)
        pos_sem_list = pd.read_pickle(pos_sem_path)
        neg_sem_list = pd.read_pickle(neg_sem_path)
        aspect_label_list = []
        pos_sem_label_list = []
        neg_sem_label_list = []
        word_list = list(words)
        for w in word_list:
            aspect_label_list.append( 1 if w in aspect_list else 0 )
            pos_sem_label_list.append( 1 if w in pos_sem_list else 0 )
            neg_sem_label_list.append( 1 if w in neg_sem_list else 0 )
        train_data = pd.DataFrame()
        train_data['word'] = word_list
        train_data['aspect'] = aspect_label_list
        train_data['pos_sem'] = pos_sem_label_list
        train_data['neg_sem'] = neg_sem_label_list
        train_data.to_pickle( path )
        return train_data

if __name__ == "__main__":
    train_data = make_train_data()
    print(train_data.head())
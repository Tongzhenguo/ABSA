import pandas as pd

data = pd.read_excel('../data/BDCI2017-+泰一指尚+-测试数据（非完整数据）.xlsx',sheetname='Sheet5')
t3_cache_path = '../cache/t3_list.pkl'

###正负分类比重
def combine( x ):
    x = str(x)
    sum_pos = 0
    sum_neg = 0
    for i in x.split(';'):
        if i =='1':
            sum_pos +=  1
        if i == '-1':
            sum_neg += 1
    return '%s_%s' % (sum_pos,sum_neg)

# data['combine_pos_neg'] = data['sentiment_anls-情感正负面'].apply( combine )
# data['pos'] = data['combine_pos_neg'].apply(lambda x:int(x.split('_')[0]))
# data['neg'] = data['combine_pos_neg'].apply(lambda x:int(x.split('_')[1]))
# sum_pos,sum_neg = sum( data['pos'].values ),sum( data['neg'].values )
# print('sum_pos / sum_neg: %s/%s' % (sum_pos,sum_neg)) #sum_pos / sum_neg: 12065/8723

#定义自己的抽取器,提取(主题词,情感词,极性)
# t3_list = []
# for i in range( len(data) ):
#     row = data.iloc[i]
#     row_id,content,aspect,semtiment_words,semtiment_anls = \
#         str(row['row_id']),str(row['content-评论内容']),str(row['theme-主题']),str(row['sentiment_word-情感关键词']),str(row['sentiment_anls-情感正负面'])
#     for j in range( len(aspect[:-1].split(';')) ):
#         aspect_word,semtiment_word,semtiment_anl = aspect[:-1].split(';')[j], semtiment_words[:-1].split(';')[j], semtiment_anls[:-1].split(';')[j]
#         t3_list.append( ( aspect_word,semtiment_word,semtiment_anl ) )
# pd.to_pickle( t3_list,t3_cache_path )
# print( t3_list[:3] )
# print( t3_list[-3:] )

#根据抽取结果,添加到分词库，主题库,正向情感词，负向情感词
# import jieba
# print( len(jieba.dt.FREQ) )
# aspect_list = []
# pos_semtiment_words_list = []
# neg_semtiment_words_list = []
# t3_list = pd.read_pickle( t3_cache_path )
# for aspect_word,semtiment_word,semtiment_anl in t3_list:
#     if aspect_word not in jieba.dt.FREQ:
#         jieba.suggest_freq(aspect_word)
#     freq = jieba.dt.FREQ.get(aspect_word, 0) + 1
#     jieba.add_word(aspect_word, tag='n', freq=freq)
#
#     if semtiment_word not in jieba.dt.FREQ:
#         jieba.suggest_freq(semtiment_word)
#     freq = jieba.dt.FREQ.get(semtiment_word, 0) + 1
#     jieba.add_word(semtiment_word, tag='n', freq=freq )
#
#     if aspect_word not in aspect_list:
#         aspect_list.append( aspect_word )
#     if semtiment_anl == '1':
#         pos_semtiment_words_list.append( semtiment_word )
#     if semtiment_anl == '-1':
#         neg_semtiment_words_list.append( semtiment_word )
# print( len(jieba.dt.FREQ) )
# pd.to_pickle( aspect_list,'../cache/aspect_list.pkl' )
# pd.to_pickle( pos_semtiment_words_list,'../cache/pos_semtiment_words_list.pkl' )
# pd.to_pickle( neg_semtiment_words_list,'../cache/neg_semtiment_words_list.pkl' )
#
# print( list(jieba.cut('冷冻效果好。刚开机！！一个小时里面就结霜啦！空间也很大！比我想象的要好！！！到货也很快！！！',HMM=False,cut_all=False)) )




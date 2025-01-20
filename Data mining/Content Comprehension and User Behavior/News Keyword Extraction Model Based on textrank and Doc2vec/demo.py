import os
os.chdir("E:\PythonNotebook\Top方案\文本关键词抽取")
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.preprocessing import MinMaxScaler
import re
import pickle
from operator import itemgetter
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import gc
import math
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import skew, kurtosis
from collections import Counter
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models import Word2Vec

pd.options.display.max_rows = 700
all_docs = pd.read_csv('prepross_all_docs.csv')

## 搜狗+百度词典 深蓝词典转换
jieba.load_userdict('./字典/明星.txt')
jieba.load_userdict('./字典/实体名词.txt')
jieba.load_userdict('./字典/歌手.txt')
jieba.load_userdict('./字典/动漫.txt')
jieba.load_userdict('./字典/电影.txt')
jieba.load_userdict('./字典/电视剧.txt')
jieba.load_userdict('./字典/流行歌.txt')
jieba.load_userdict('./字典/创造101.txt')
jieba.load_userdict('./字典/百度明星.txt')
jieba.load_userdict('./字典/美食.txt')
jieba.load_userdict('./字典/FIFA.txt')
jieba.load_userdict('./字典/NBA.txt')
jieba.load_userdict('./字典/网络流行新词.txt')
jieba.load_userdict('./字典/显卡.txt')

## 爬取漫漫看网站和百度热点上面的词条
jieba.load_userdict('./字典/漫漫看_明星.txt')
jieba.load_userdict('./字典/百度热点人物+手机+软件.txt')
jieba.load_userdict('./字典/自定义词典.txt')

## 实体名词抽取之后的结果 有一定的人工过滤 
## origin_zimu 这个只是把英文的组织名过滤出来
jieba.load_userdict('./字典/person.txt')
jieba.load_userdict('./字典/origin_zimu.txt')

## 第一个是所有《》里面出现的实体名词
## 后者是本地测试集的关键词加上了 
jieba.load_userdict('./字典/出现的作品名字.txt')
jieba.load_userdict('./字典/val_keywords.txt')

## 网上随便找的停用词合集
jieba.analyse.set_stop_words('./stopword.txt')

val = pd.read_csv('train_docs_keywords.txt', sep='\t', header=None)
val.columns = ['id', 'kw']
val.kw = val.kw.apply(lambda x: x.split(','))

all_docs = pd.read_csv('prepross_all_docs.csv')
all_docs = all_docs[['id','title','content','title_cut','content_cut','first_sentence','other_sentence','last_sentence','first_sentence_reg']]


## 历史作品名字 作为下面的一个特征
TV = []
with open('./字典/出现的作品名字.txt', 'r', encoding='utf-8') as f:
    for word in f.readlines():
        TV.append(word.strip())

## 这是根据本数据集算的idf文件
idf = {}
with open('my_idf.txt', 'r', encoding='utf-8') as f:
    for i in f.readlines():
        if len(i.strip().split()) == 2:
            v = i.strip().split()
            idf[v[0]] = float(v[1])
            
def evaluate(df):
    def get_score(x):
        score = 0
        if x['label1'] in x['kw']:
            score += 0.5
        if x['label2'] in x['kw']:
            score += 0.5 
        return score
    
    pred = df[df.id.isin(val.id)]
    tmp  = pd.merge(pred, val, on='id', how='left')
    tmp['score'] = tmp.apply(get_score, axis=1)
    print('Score: ',tmp.score.sum())
    return tmp
## classes_doc2vec.npy 文件是先算DOC2VEC向量 然后用Kmeans简单聚成10类
classes = np.load('classes_doc2vec.npy')

all_docs['classes'] = classes
doc2vec_model = Doc2Vec.load('doc2vec.model')

word2vec_model = Word2Vec.load('word2vec.model')

wv= word2vec_model.wv
def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))


def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())
## 后面加大窗口和迭代又算了一次word2vec 模型 主要是用来算候选关键词之间的相似度

word2vec_model_256 = Word2Vec.load('word2vec_iter10_sh_1_hs_1_win_10.model')
all_docs['idx'] = all_docs.index.values

def get_train_df(df, train=True):
    res = []
    for index in tqdm(df.index):
        #遍历获取每一条数据
        x = df.loc[index]
        # TF-IDF候选关键词结果
        first_sentence_reg = ' '.join(x['first_sentence_reg'])
        
        # 对文章进行手动加权
        text = 19*(str(x['title_cut'])+'。')+ 3*(str(x['first_sentence'])+'。') + 1*(str(x['other_sentence'])+'。')+\
                3*(str(x['last_sentence'])+ '。') + 7*str(first_sentence_reg)
        # TF-IDF候选关键词结果
        # 注：原作者更改了结巴分词中的代码，参数allowpPOS实际是不允许出现的词性 即allowPOS = NotAllowPOS
        jieba_tags = jieba.analyse.extract_tags(sentence=text, topK=20, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'), withWeight=True,\
                                          withFlag=True)
        #获得extract_tags中所有结果
        tags = []
        cixing = []
        weight = []
        len_tags = []
        for tag in jieba_tags:
            tags.append(tag[0].word)
            cixing.append(tag[0].flag)
            weight.append(tag[1])
        
        # 分隔符：用来切分句子用
        sentence_delimiters = re.compile(u'[。？！；!?]')
        # 切分句子
        sentences =[i for i in sentence_delimiters.split(text) if i != '']
        # 包含句子数量
        num_sen = len(sentences)
        # 获得当前文本所有词语
        words = []
        num_words = 0
        for sen in sentences:
            cut = jieba.lcut(sen)
            words.append(cut)
            num_words += len(cut)
        for i in range(len(tags)):
            len_tags.append(len(tags[i]))
            
        new_tags = tags
        new_weight = weight
        new_cixing = cixing
        
        
            
        ## 位置特征： 1. 是否出现在标题 2.是否出现在第一句 3.是否出现在最后一句 4.出现在正文中间部分
        occur_in_title = np.zeros(len(new_tags))
        occur_in_first_sentence = np.zeros(len(new_tags))
        occur_in_last_sentence = np.zeros(len(new_tags))
        occur_in_other_sentence = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            if new_tags[i] in x['title_cut']:
                occur_in_title[i] = 1
            if new_tags[i] in x['first_sentence']:
                occur_in_first_sentence[i] = 1
            if new_tags[i] in x['last_sentence']:
                occur_in_last_sentence[i] = 1
            try:
                if new_tags[i] in x['other_sentence']:
                    occur_in_other_sentence[i] = 1
            except:
                occur_in_other_sentence[i] = 1
                print (new_tags[i])

                    
        
        
        ## 共现矩阵及相关统计特征,例如均值、方差、偏度等 
        num_tags = len(new_tags)
        #初始化共现矩阵
        arr = np.zeros((num_tags, num_tags))
        # 计算出现个数
        for i in range(num_tags):
            for j in range(i+1, num_tags):
                count = 0
                for word in words:
                    if new_tags[i] in word and new_tags[j] in word:
                        count += 1
                arr[i, j] = count
                arr[j, i] = count
        # 计算统计特征
        ske = stats.skew(arr)
        var_gongxian = np.zeros(len(new_tags))
        kurt_gongxian = np.zeros(len(new_tags))
        diff_min_gongxian = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            var_gongxian[i] = np.var(arr[i])
            kurt_gongxian[i] = stats.kurtosis(arr[i])
            diff_sim = np.diff(arr[i])
            if len(diff_sim) > 0:
                diff_min_gongxian[i] = np.min(diff_sim)

                
        ## textrank特征，跟pagerank原理类似，如果一个单词出现在很多单词后面的话，那么说明这个单词比较重要，
        ## 一个TextRank值很高的单词后面跟着的一个单词，那么这个单词的TextRank值会相应地因此而提高
        textrank_tags = dict(jieba.analyse.textrank(sentence=text, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'), withWeight=True))
        
        textrank = []
        for tag in new_tags:
            if tag in textrank_tags:
                textrank.append(textrank_tags[tag])
            else:
                textrank.append(0)
        # 得到所有词，list格式
        all_words = np.concatenate(words).tolist()
        
        ## 词频
        tf = []
        for tag in new_tags:
            tf.append(all_words.count(tag))
        tf = np.array(tf)
        
        ## hf: 头词频，文本内容前1/4候选词词频
        hf = []
        head = len(words) // 4 + 1
        head_words = np.concatenate(words[:head]).tolist()
        for tag in new_tags:
            hf.append(head_words.count(tag))
        
        ## has_num：是否包含数字
        ## has_eng: 是否包含字母
        def hasNumbers(inputString):
            return bool(re.search(r'\d', inputString))
        def hasEnglish(inputString):
            return bool(re.search(r'[a-zA-Z]', inputString))
        has_num = []
        has_eng = []
        for tag in new_tags:
            if hasNumbers(tag):
                has_num.append(1)
            else:
                has_num.append(0)
            if hasEnglish(tag):
                has_eng.append(1)
            else:
                has_eng.append(0)
                
        ## is_TV:是否为作品名称
        is_TV = []
        for tag in new_tags:
            if tag in TV:
                is_TV.append(1)
            else:
                is_TV.append(0)
                
        ## idf: 用训练集跑出的逆词频
        v_idf = []
        for tag in new_tags:
            v_idf.append(idf.get(tag, 0)) 
        
        ## 计算文本相似度，这里直接用doc2vec跟每个单词的word2vec做比较
        ## sim: 余弦相似度
        ## sim_euc：欧氏距离
        default = np.zeros(100)
        doc_vec = doc2vec_model.docvecs.vectors_docs[x['idx']]
        sim = []
        sim_euc = []
        for tag in new_tags:
            if tag in wv:
                sim.append(Cosine(wv[tag], doc_vec))
                sim_euc.append(Euclidean(wv[tag], doc_vec))
            else:
                sim.append(Cosine(default, doc_vec))
                sim_euc.append(Euclidean(default, doc_vec))
                
        ## 关键词所在句子长度，记录为列表，然后算统计特征 
        mean_l2 = np.zeros(len(new_tags))
        max_l2 = np.zeros(len(new_tags))
        min_l2 = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            tmp = []
            for word in words:
                if new_tags[i] in word:
                    tmp.append(len(word))
            if len(tmp) > 0:
                mean_l2[i] = np.mean(tmp)
                max_l2[i] = np.max(tmp)
                min_l2[i] = np.min(tmp)
                
        ## 关键词所在位置，记录为列表，然后算统计特征 
        min_pos = [np.NaN for _ in range(len(new_tags))]
        diff_min_pos_bili = [np.NaN for _ in range(len(new_tags))]
        diff_kurt_pos_bili = [np.NaN for _ in range(len(new_tags))]
        
        for i in range(len(new_tags)):
            # 得到当前候选标签出现的所有位置
            pos = [a for a in range(len(all_words)) if all_words[a] == new_tags[i]]
            # 计算相对位置
            pos_bili = np.array(pos) / len(all_words)
            
            if len(pos) > 0:
                min_pos[i] = np.min(pos)
                # 差分特征
                diff_pos = np.diff(pos)
                diff_pos_bili = np.diff(pos_bili)
                if len(diff_pos) > 0:
                    diff_min_pos_bili[i] = np.min(diff_pos_bili)
                    diff_kurt_pos_bili[i] = stats.kurtosis(diff_pos_bili)   
        ## 候选关键词之间的相似度 word2vec gensim 窗口默认 迭代默认 向量长度100
        ## sim_tags_arr：相似度矩阵
        sim_tags_arr = np.zeros((len(new_tags), len(new_tags)))
        for i in range(len(new_tags)):
            for j in range(i+1, len(new_tags)):
                if new_tags[i] in wv and new_tags[j] in wv:
                    sim_tags_arr[i, j] = word2vec_model.similarity(new_tags[i], new_tags[j])
                    sim_tags_arr[j, i] = sim_tags_arr[i, j]
        # 计算当前次和其他词平均结果
        mean_sim_tags = np.zeros(len(new_tags))
        diff_mean_sim_tags = np.zeros(len(new_tags))     
        for i in range(len(new_tags)):
            mean_sim_tags[i] = np.mean(sim_tags_arr[i])
            diff_sim = np.diff(sim_tags_arr[i])
            if len(diff_sim) > 0:
                diff_mean_sim_tags[i] = np.mean(diff_sim)

        ## 候选关键词之间的相似度 word2vec gensim 窗口10 迭代10 向量长度256 
    
        sim_tags_arr_255 = np.zeros((len(new_tags), len(new_tags)))
        for i in range(len(new_tags)):
            for j in range(i+1, len(new_tags)):
                if new_tags[i] in word2vec_model_256 and new_tags[j] in word2vec_model_256:
                    sim_tags_arr_255[i, j] = word2vec_model_256.similarity(new_tags[i], new_tags[j])
                    sim_tags_arr_255[j, i] = sim_tags_arr_255[i, j]
        ## label 训练集打标签
        if train:
            label = []
            for tag in new_tags:
                if tag in x.kw:
                    label.append(1)
                else:
                    label.append(0)
                    
        ## 不同词性的比例
        cixing_counter = Counter(new_cixing)
        
        fea = pd.DataFrame()
        fea['id'] = [x['id'] for _ in range(len(new_tags))]
        fea['tags'] = new_tags
        fea['cixing'] = new_cixing


        fea['tfidf'] = new_weight
        fea['ske'] = ske
        
        fea['occur_in_title'] = occur_in_title
        fea['occur_in_first_sentence'] = occur_in_first_sentence
        fea['occur_in_last_sentence'] = occur_in_last_sentence
        fea['occur_in_other_sentence'] = occur_in_other_sentence
        fea['len_tags'] = len_tags
        fea['num_tags'] = num_tags
        fea['num_words'] = num_words
        fea['num_sen'] = num_sen
        fea['classes'] = x['classes']

        fea['len_text'] = len(x['title_cut'] + x['content_cut'])
        fea['textrank'] = textrank
        fea['word_count'] = tf
        fea['tf'] = tf / num_words
        fea['num_head_words'] = len(head_words)
        fea['head_word_count'] = hf
        fea['hf'] = np.array(hf) / len(head_words)
        fea['pr'] = tf / tf.sum()
        fea['has_num'] = has_num
        fea['has_eng'] = has_eng
        fea['is_TV'] = is_TV
        fea['idf'] = v_idf
        fea['sim'] = sim
        fea['sim_euc'] = sim_euc

        fea['mean_l2'] = mean_l2
        fea['meaxl2'] = max_l2
        fea['min_l2'] = min_l2
        
        fea['min_pos'] = min_pos
        fea['diff_min_pos_bili'] = diff_min_pos_bili
        fea['diff_kurt_pos_bili'] = diff_kurt_pos_bili
    
        #fea['diff_max_min_sen_pos'] = diff_max_min_sen_pos
        #fea['diff_var_sen_pos_bili'] = diff_var_sen_pos_bili

        fea['mean_sim_tags'] = mean_sim_tags
        fea['diff_mean_sim_tags'] = diff_mean_sim_tags

        #fea['kurt_sim_tags_256'] = kurt_sim_tags_256
        #fea['diff_max_min_sim_tags_256'] = diff_max_min_sim_tags_256
        fea['var_gongxian'] = var_gongxian
        fea['kurt_gongxian'] = kurt_gongxian
        fea['diff_min_gongxian'] = diff_min_gongxian
        
        ## 当前文本候选关键词词性比例
        for c in ['x', 'nz', 'l', 'n', 'v', 'ns', 'j', 'a', 'vn', 'nr', 'eng', 'nrt',
                  't', 'z', 'i', 'b', 'o', 'nt', 'vd', 'c', 's', 'nrfg', 'mq', 'rz',
                  'e', 'y', 'an', 'rr']:
            fea['cixing_{}_num'.format(c)] = cixing_counter[c]
            fea['cixing_{}_bili'.format(c)] = cixing_counter[c] / (len(new_cixing) + 1)

        if train:
            fea['label'] = label
        res.append(fea)
    return res
train_doc = pd.merge(val, all_docs, on='id', how='left')
res = get_train_df(train_doc, train=True)
train_df = pd.concat(res, axis=0).reset_index(drop=True)
"""
res = get_train_df(all_docs, train=False)
test_df = pd.concat(res, axis=0).reset_index(drop=True)
"""
train_df.to_csv('my_train_df.csv', index=False)
#test_df.to_csv('my_train_df.csv', index=False)

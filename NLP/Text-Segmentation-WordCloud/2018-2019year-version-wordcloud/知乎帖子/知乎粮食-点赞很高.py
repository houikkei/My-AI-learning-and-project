import wordcloud
import jieba
import imageio
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import pandas as pd
import re

a1=pd.read_csv('点赞量很高的评论.csv',encoding='utf_8_sig')

del a1['voteup_count']
del a1['headline']
del a1['created_time']
del a1['updated_time']
del a1['comment_count']
del a1['thanks_count']
del a1['excerpt']
del a1['label']
del a1['gender']
a1 = pd.DataFrame(a1, columns=['content'])
a1.to_csv("a1.txt",encoding='utf_8_sig')

filename = r"a1.txt"
inf = pd.read_csv(filename)
inf_need = inf['content']
inf_need = inf_need.values
inf_txt = ''
for i in range(0, 774):
    inf_txt = inf_txt + str(inf_need[i])

inf_txt_CN = re.findall(r'[\u4e00-\u9fa5]', inf_txt)
string = ''
for i in range(0, 110000):
    string = string + inf_txt_CN[i]

with open(r"a1.txt", "w", encoding='utf-8') as f:
    f.write(string)
    f.close()

## 设置禁用词
stopwords=set()
stopwords.add('我们')
stopwords.add('已经')
stopwords.add('还是')
stopwords.add('但是')
stopwords.add('应该')
stopwords.add('真的')
stopwords.add('什么')
stopwords.add('那么')
stopwords.add('其实')
stopwords.add('就是')
stopwords.add('没有')
stopwords.add('自己')
stopwords.add('一个')
stopwords.add('这些')
stopwords.add('因为')
stopwords.add('只是')
stopwords.add('这种')
stopwords.add('如果')
stopwords.add('不是')
stopwords.add('他们')
stopwords.add('这个')
stopwords.add('这么')
stopwords.add('怎么')
stopwords.add('比如')
stopwords.add('当然')
stopwords.add('怎么')
stopwords.add('而是')
stopwords.add('或者')
stopwords.add('比如')
stopwords.add('你们')
stopwords.add('一下')
stopwords.add('是不是')
stopwords.add('等等')
stopwords.add('多少')
stopwords.add('如何')
stopwords.add('一天')
stopwords.add('一点')
stopwords.add('的话')
stopwords.add('所以')
stopwords.add('一些')

mk = imageio.imread('1.jpg')
w = wordcloud.WordCloud(width=1000,  ## scale 调整清晰度
                        height=700,
                        background_color='white',
                        mask=mk,
                        stopwords=stopwords,
                        font_path='msyh.ttc',
                       scale=10)
f = open('a1.txt',encoding='utf-8')
txt = f.read()
article_contents = ""
    #使用jieba进行分词
words = jieba.cut(txt,cut_all=False)
for word in words:
        #使用空格来分割词
        article_contents += word+" "

string=""
string = "".join(article_contents)
w.generate(string)
w.to_file('点赞很高.png')
f.close()


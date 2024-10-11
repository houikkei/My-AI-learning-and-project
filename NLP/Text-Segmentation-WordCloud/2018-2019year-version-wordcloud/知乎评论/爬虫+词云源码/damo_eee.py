from zhihu_oauth import ZhihuClient
import time
import random
import pandas as pd

client=ZhihuClient()
client.load_token('token.pkl')

id=325177150
question=client.question(id)
answers=question.answers
rows=[]
num=1
for ans in answers:
    if num<=question.answer_count:
        print("No.%d"%num,ans.author.name,ans.voteup_count,ans.author.headline,ans.author.gender,ans.created_time,ans.comment_count)
        dic={
            'No.':num,
            'author':ans.author.name,
            'voteup_count':ans.voteup_count,
            'content':ans.content,
            'headline':ans.author.headline,
            'gender':ans.author.gender,
            'created_time':ans.created_time,
            'updated_time':ans.updated_time,
            'comment_count':ans.comment_count,
            'thanks_count':ans.thanks_count,
            'excerpt':ans.excerpt
            }
        rows.append(dic)
        num+=1

df=pd.DataFrame(rows)
#print(df)
df.to_csv("如何看待因美国禁令华为手机将被暂停谷歌移动服务支持？将产生多大范围的影响？.csv",encoding='utf-8',index=False)#sep='\t',
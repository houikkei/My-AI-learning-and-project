from zhihu_oauth import ZhihuClient
from zhihu_oauth.exception import NeedCaptchaException
import time
import random
import pandas as pd
client = ZhihuClient()

try:
    client.login('15600679719', 'xuebi123')
except NeedCaptchaException:
    # 保存验证码并提示输入，重新登录
    with open('a.gif', 'wb') as f:
        f.write(client.get_captcha())
    captcha = input('please input captcha:')  # 验证码在文件夹中，手动输入
    client.login('15600679719', 'xuebi123', captcha)

client.save_token('token.pkl')

id=383352936
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
df.to_csv("疫情尚未结束，多国禁止粮食出口，老百姓需要囤积粮食吗？.csv",encoding='utf-8',index=False)#sep='\t',
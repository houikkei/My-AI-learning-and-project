import requests
import re
import json


def main(page):
    url = 'http://bang.dangdang.com/books/fivestars/1-' + str(page)
    html = request_dandan(url)
    items = parse_result(html)

    for item in items:
        write_item_to_file(item)


def request_dandan(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None


def parse_result(html):
    pattern = re.compile('<li>.*?list_num.*?(\d+).</div>.*?<img src="(.*?)".*?class="name".*?title="(.*?)">.*?class="star">.*?target="_blank">(.*?)</a>.*?class="tuijian">(.*?)</span>.*?class="publisher_info">.*?target="_blank">(.*?)</a>.*?class="publisher_info".*?<span>(.*?)</span>.*?target="_blank">(.*?)</a>.*?class="biaosheng">.*?<span>(.*?)</span></div>.*?<p><span\sclass="price_n">&yen;(.*?)</span>.*?span\sclass="price_r">&yen;(.*?)</span>.*?</li>',re.S)
    items = re.findall(pattern,html)
    for item in items:
        yield {
            
            '排名': item[0],
            '图片': item[1],
            '书名': item[2],
            '评论数': item[3],
            '推荐度': item[4],
            '作者': item[5],
            '出版时间': item[6],
            '出版社': item[7],
            '五星评分次数': item[8],
            '价格': item[9],
            '原价': item[10]
            
        }


def write_item_to_file(item):
    print('开始写入数据 ===> ' + str(item))
    with open('book.txt', 'a', encoding='UTF-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')





if __name__ == '__main__':
    for i in range(1, 26):
        main(i)



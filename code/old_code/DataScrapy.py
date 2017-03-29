#!/usr/bin/env python
# coding=utf-8

"""
    可用的搜索引擎：搜狗新闻搜索（可以给出原网址链接）和百度搜索（百度的链接跳转被加密，不容易破解）
    可爬取评论的新闻网站：腾讯新闻网站，可以通过url匹配的方法获取新闻搜索的评论
"""

import sys
import json
import time
import random
import urllib
import urllib2
from argparse import ArgumentParser
import re

reload(sys)
sys.setdefaultencoding('utf-8')


class DataScrapy:

    def __init__(self, keyword=''):
        self.keyword = keyword
        self.websites = ['https://www.sogou.com/sogou?\
site=news.qq.com&query={0}&\
pid=sogou-wsse-b58ac8403eb9cf17-0004&\
sourceid=&idx=f&idx=f'.format(keyword),
                         'http://www.baidu.com/s?\
wd={0}%%20site%%3Anews.qq.com&\
rsv_spt=1&rsv_iqid=0x897905fe00080c14&issp=1&f=3&\
rsv_bp=0&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_enter=1&\
rsv_sug3=14&rsv_sug1=13&rsv_sug7=101&rsv_sug2=0&rsp=0&\
rsv_sug9=es_0_1&inputT=4758&rsv_sug4=4758&rsv_sug=9'.format(keyword), ]
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_3) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.54 Safari/536.5'
        self.headers = {'User-Agent': user_agent}
        self.filepath = '../data/urllist.txt'

    def getUrlList(self, engine, pagenum):
        fw = open(self.filepath, 'w')
        if engine == 'sogou':
            url = self.websites[0]
            pattern = u'推荐您在' + r'.*?<a target="_blank" href="(http://news\.qq\.com/.*?\.htm)'

        elif engine == 'baidu':
            url = self.websites[1]
            pattern = r'<div class="f13"><a target="_blank" href="(.*?)"'
        else:
            print '[Error]: You must choose one search engine(baidu or sogou)!'
            fw.close()
            return

        # print url
        page = 1
        useful_link = 0
        url_set = set()
        while useful_link < pagenum:
            page_url = r'&page={0}'.format(page)
            url_tmp = url
            url_tmp += page_url
            # print url_tmp
            page += 1
            if page > 30:
                break
            req = urllib2.Request(url_tmp.encode('gbk'), headers=self.headers)
            res = urllib2.urlopen(req)
            content = res.read().decode('utf-8')
            # print content
            news_list = re.findall(pattern, content, re.S)
            useful_link += len(news_list)
            for each_web in news_list:
                date = re.findall(r'(20\d{6})', each_web)
                if len(date)==1 and date[0] > '20150101':
                    url_set.add(each_web)
                # req = urllib2.Request(each_web, headers = self.headers)
                # res = urllib2.urlopen(req)
                # print res.read()
            time.sleep(5)
        for each_web in url_set:
            fw.write(each_web + '\n')
            print each_web
        fw.close()
    def ReviewsReader(self, url, reviews, writetype='w'):
        fw = open(reviews, writetype)
        comment_num = 0
        comment_url = set()
        # for url in url_list:
        req = urllib2.Request(url, headers=self.headers)
        res = urllib2.urlopen(req)
        content = res.read().decode('gbk', 'ignore')

        pattern_newurl = r'(http://\w+\.qq\.com/a/\d+/\d+\.htm)'
        match_newurl = re.findall(pattern_newurl, content, re.S)
        match_newurl = set(match_newurl)
        print 'The new urls: '
        print match_newurl
        comment_url |= match_newurl

        pattern_cmtid = r'cmt_id = (\d+);|aid: "(\d+)"'
        match = re.findall(pattern_cmtid, content, re.S)
        try:
            match = match[0]
            if match[0] != '':
                match = match[0]
            elif match[1] != '':
                match = match[1]
            else:
                return comment_num, comment_url
        except IndexError:
            return comment_num, comment_url
        time.sleep(1)
        lastid = '0'
        while True:
            cmt_url = r'http://coral.qq.com/article/{0}/comment?commentid={1}&reqnum=20&tag=&callback=mainComment&_=1389623278900'.format(match, lastid)
            req = urllib2.Request(cmt_url, headers=self.headers)
            res = urllib2.urlopen(req)
            content = res.read().decode()
            comments = re.findall(r'mainComment\((.*)\)', content, re.S)[0]
            comments = json.loads(comments)
            comments = comments['data']['commentid']
            
            if len(comments) != 0:
                comment_num += len(comments)
                for comment in comments:
                    fw.write(comment['content'] + '\n')
                    # print comment['content']
                if len(comments) == 20:
                    lastid = comments[19]['id']
                else:
                    break
            else:
                break
            time.sleep(2)

        fw.close()
        return comment_num, comment_url

    def ReviewsScrapy(self, begin_url, reviews='../data/reviews.txt', reviewnum=5000):
        req = urllib2.Request(begin_url, headers=self.headers)
        res = urllib2.urlopen(req)
        content = res.read().decode('gbk', 'ignore')
        # print content

        total_list = set()
        url_list1 = set()
        url_list2 = set()

        pattern = r'(http://\w+\.qq\.com/a/\d+/\d+\.htm)'
        urls = re.findall(pattern, content, re.S)
        url_list1 = set(urls)
        print 'The start url number is: {0}'.format(len(url_list1))
        print url_list1
        total = 0
        max_up = reviewnum
        while len(url_list1) != 0:
            total_list |= url_list1
            url_list1 = list(url_list1)
            random.shuffle(url_list1)
            for url in url_list1:
                print 'Current url reading: {0}'.format(url)
                try:
                    review_num, new_url_list = self.ReviewsReader(url=url, reviews=reviews, writetype='a')
                    url_list2 |= new_url_list
                    total += review_num
                    print 'I have got {0} reviews. {1}% Finished...'.format(total, 100*total/float(max_up))
                    if total >= max_up:
                        break
                except Exception:
                    print '[Error]: no useful information!'
                    continue
            url_list1 = url_list2 - total_list
            url_list2 = set()
            time.sleep(1)



if __name__ == '__main__':
    arg_parser = ArgumentParser(description='args')
    arg_parser.add_argument('--pagenum', dest='pagenum')
    args = arg_parser.parse_args()
    ds = DataScrapy(u'高考名额分配')
    # ds.getUrlList('sogou', int(args.pagenum))
    # ds.ReviewsReader()
    ds.ReviewsScrapy(r'http://news.qq.com/paihangz.htm', '../data/test_reviews.txt', 2000)

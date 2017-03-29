#/bin/bash
#-*- encoding:utf-8 -*-


PYTHON='python'

# 预处理文件、分词
# $PYTHON Word2VecTestOfWiki.py \
# ../data/wiki/zhwiki-latest-pages-articles.xml.bz2 \
# ../data/wiki/wiki.zh.text

# if [[ $? -eq 0 ]]; then
# 	echo -e "文件预处理成功！"
# else
# 	exit(1)
# fi

# 繁体转简体
# opencc -i ../data/wiki/wiki.zh.text -o \
# ../data/wiki/wiki.zh.text.jian -c /usr/local/share/opencc/t2s.json

# echo $?
# if [[ $? -eq 0 ]]; then
# 	echo -e "繁体转简体成功！"
# else
# 	exit(1)
# fi

# # 分词
# mecab -d ../data/ -O wakati ../data/wiki/wiki.zh.text.jian -o \
# ../data/wiki/wiki.zh.text.jian.seg -b 10000000

# 编码转换为utf8
# iconv -c -t UTF-8 < ../data/wiki/wiki.zh.text.jian > \
# ../data/wiki/wiki.zh.text.jian.utf-8

# echo $?
# if [[ $? -eq 0 ]]; then
# 	echo -e "编码转换成功！"
# else
# 	exit(1)
# fi

# 进行word2vec处理
$PYTHON Word2VecTestOfWiki.py \
../data/wiki/wiki.zh.text.jian.utf-8 \
../data/wiki/wiki.zh.text.model \
../data/wiki/wiki.zh.text.vector

echo $?
if [[ $? == 0 ]]; then
	echo -e "Word2Vec模型训练成功！"
else
	exit(1)
fi


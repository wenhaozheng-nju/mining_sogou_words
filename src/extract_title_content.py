#!/usr/bin/env python
# coding=utf-8
import re
import jieba,codecs
import jieba.posseg as pseg
from gensim.models import Word2Vec

tags = ['n','v','a','r','d','j','t','g','f','i','m','s']
re_h=re.compile('</?\w+[^>]*>')
my_dir = '../result/'

def handle_one_line(my_str):
    if my_str.find('<contenttitle>') >=0 or my_str.find('<content>') >= 0:
        my_str = re_h.sub('',my_str)
        #seg_list = jieba.cut(my_str,cut_all=True)
        seg_list = pseg.cut(my_str)
        words = [word for (word,flag) in seg_list if flag[0].lower() in tags]
        sents_file.write(' '.join(words)+"\n")
        
class MySentences(object):
    def __init__(self,filename):
        self.filename = filename
        self.count = 0

    def __iter__(self):
        print self.count
        self.count += 1
        for line in codecs.open(self.filename,'r','utf8'):
            #print "hehe"
            yield line.split()

def my_word2vec(my_file):
    #sents = MySentences(my_file)
    #model = Word2Vec(min_count=2,size=500,sample=1e-4,iter=20,negative=10)
    #model.build_vocab(sents)
    #sents = MySentences(my_file)
    #model.train(sents)
    sents = MySentences(my_file)
    #sents = []
    #for line in codecs.open(my_file,'r','utf8'):
    #    sents.append(line.split())
    model = Word2Vec(sents,min_count=2,size=100,sample=1e-5,iter=20,workers=3,negative=15)
    model.save(my_dir + 'word2vec.model')

#sents_file = codecs.open('sents_file.data','w','utf8')
#for line in codecs.open('./news_sohusite_xml.dat','r','gb18030'):
#    handle_one_line(line)
my_word2vec(my_dir+'sents_file.data')

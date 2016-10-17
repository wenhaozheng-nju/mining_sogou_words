#!/usr/bin/env python
# coding=utf-8

import jieba.posseg as pseg
import codecs

def cut_words(sent):
    words = pseg.cut(sent)
    n_words = []
    for word, flag in words:
        if flag[0] == 'n':
            n_words.append(word)
    return n_words

def read_file(filename,dst_file):
    dst_file = codecs.open(dst_file,"w","gb18030")
    for line in codecs.open(filename,"r",'gb18030'):
        eles = line.split()
        n_words = []
        for sent in eles[4:]:
            n_words += cut_words(sent)
        dst_file.write(" ".join(eles[1:4]+n_words)+"\n")

def read_file_2(filename,dst_file):
    dst_file = codecs.open(dst_file,"w","gb18030")
    for line in codecs.open(filename,"r",'gb18030'):
        eles = line.split()
        n_words = []
        for sent in eles[1:]:
            n_words += cut_words(sent)
        dst_file.write(eles[0]+" "+" ".join(n_words)+"\n")

if __name__ == '__main__':
    read_file('./user_tag_query.2W.TRAIN','./whole_train.csv')
    #read_file_2("./user_tag_query.2W.TEST",'./test.csv')   



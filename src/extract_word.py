#!/usr/bin/env python
# coding=utf-8
import codecs


def extract(filename,write_file):
    write_file = codecs.open(write_file,"w",'gb18030')
    for line in codecs.open(filename,"r",'gb18030'):
        eles = line.split()
        sents = eles[4:]
        write_file.write(','.join(sents)+'\n')

extract('./user_tag_query.2W.TRAIN','./train_sents.txt')
extract('./user_tag_query.2W.TEST','./test_sents.txt')


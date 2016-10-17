#!/usr/bin/env python
# coding=utf-8
import codecs

word_dict = {}
#selected_tag = ['n','v','i','j','a','d','t','T','z']
selected_tag = ['n']

def read_word_dict():
    word_dict_file = codecs.open('./word_dict.txt','r','utf8')
    for line in word_dict_file:
        ele = line.split()
        if len(ele) > 1:
            word_dict[ele[0]] = ele[1]

def gen_dataset(src_file, dst_file):
    global word_dict
    dst_file = codecs.open(dst_file, 'w', 'gb18030')
    for line in codecs.open(src_file, "r", 'utf8'):
        eles = line.split()
        for ele in eles:
            #print ele
            words = ele.split("/")
            word = "".join(words[:-1])
            #if word not in word_dict:
            #    word_dict[word] = 0
            #word_dict[word] += 1
            tag = words[-1]
            if word in word_dict and word_dict[word] > 1 and (tag[0] in selected_tag):
                # word.decode('utf8').encode('gb18030')
                dst_file.write(word + "\t")
        dst_file.write("\n")
    #word_array = sorted(word_dict.iteritems(),key = lambda d:d[1],reverse=True)
    #word_dict_file = codecs.open('./word_dict.txt',"w",'utf8')
    #map(lambda x:word_dict_file.write(x[0]+" "+str(x[1])+'\n'),word_array)

def paste_column(word_file,label_file,final_file,flag):
    final_file = codecs.open(final_file,'w','gb18030')
    with codecs.open(word_file,'r','gb18030') as f1, codecs.open(label_file,'r','gb18030') as f2:
        for word_line in f1:
            label_line = f2.readline()
            if flag:
                labels = label_line.split()[:4]
            else:
                labels = label_line.split()[:1]
            final_file.write(" ".join(labels)+" "+word_line)
    
#read_word_dict()
#gen_dataset('./train_words.txt', './trainSet_nlpir.txt')
#gen_dataset('./test_words.txt', './testSet_nlpir.txt')
paste_column('./trainSet_nlpir.txt','./user_tag_query.2W.TRAIN','./train_nlpir_final.csv',True)
paste_column('./testSet_nlpir.txt','./user_tag_query.2W.TEST','./test_nlpir_final.csv',False)

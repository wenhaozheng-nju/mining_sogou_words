#!/usr/bin/env python
# coding=utf-8
from __future__ import division
import codecs

word_dict = {}
word_count = {}

def read_rec(record):
    global word_dict,word_count
    eles = record.split()
    (label1,label2,label3) = eles[:3]
    sent_len = len(eles[:3])
    for word in eles[3:]:
        if word not in word_dict:
            word_dict[word] = [[0]*7,[0]*3,[0]*7]
        word_dict[word][0][int(label1)] += 1/sent_len
        word_dict[word][1][int(label2)] += 1/sent_len
        word_dict[word][2][int(label3)] += 1/sent_len
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
        

def number2ratio(arr):
    my_sum = sum(arr)
    if my_sum == 0:
        return arr
    new_arr = [ele/my_sum for ele in arr]
    return new_arr

def rerange():
    global word_dict
    word_file = codecs.open('word_file.txt',"w","gb18030")
    word_count_file = codecs.open('word_count.txt','w','gb18030')
    for word in word_dict:
        word_dict[word][0] = number2ratio(word_dict[word][0])
        word_dict[word][1] = number2ratio(word_dict[word][1])
        word_dict[word][2] = number2ratio(word_dict[word][2])
        word_file.write(word+" label1: "+" ".join(map(str,word_dict[word][0]))+" label2: "+" ".join(map(str,word_dict[word][1]))+" label3: "+" ".join(map(str,word_dict[word][2]))+"\n")
        word_count_file.write(word+" "+str(word_count[word])+"\n")

def add_matrix(left,right,word):
    global word_count
    for i in range(3):
        #print ele
        for j in range(len(left[i])):
            if right[i][j] == 1:
                left[i][j] += 100
            else:
                left[i][j] = left[i][j] + right[i][j]
    return left

def find_max(my_list):
    sorted_list = sorted(my_list,reverse=True)
    if sorted_list[0] - sorted_list[1] > 100 and (sorted_list[0]-sorted_list[1])/sorted_list[1] > 0.5:
        return my_list.index(sorted_list[0])
    else:
        return -1


def predict_label(test_file,submission,submission_tmp):
    global word_dict,word_count
    submission = codecs.open(submission,'w','gb18030')
    submission_tmp = codecs.open(submission_tmp,'w','gb18030')
    for line in codecs.open(test_file,"r",'gb18030'):
        eles = line.split()
        ID = eles[0]
        labels = [[0]*7,[0]*3,[0]*7]
        for word in eles[3:]:
            if word in word_dict and word_count[word] > 1:
                #print "pre:",labels.dtype
                #print "word_dict:",np.array(word_dict[word]).dtype
                #print "after:",labels.dtype
                #assert(0)
                #print word
                labels = add_matrix(labels,word_dict[word],word)
        #assert(0)
        #label1 = labels[0].index(max(labels[0]))
        label1 = find_max(labels[0])
        label2 = find_max(labels[1])
        label3 = find_max(labels[2])
        #if label1 == 0:
        #    label1 = 1
        #label2 = labels[1].index(max(labels[1]))
        #if label2 == 0:
        #    label2 = 1
        #label3 = labels[2].index(max(labels[2]))
        #if label3 == 0:
        #    label3 = 5
        #submission.write(ID+" "+str(label1)+" "+str(label2)+" "+str(label3)+"\n")
        submission.write(ID+" "+str(label1)+" "+str(label2)+" "+str(label3)+"\n")
        labels[0] = map(str,labels[0])
        labels[1] = map(str,labels[1])
        labels[2] = map(str,labels[2])
        submission_tmp.write(ID+","+" ".join(labels[0])+","+" ".join(labels[1])+","+" ".join(labels[2])+"\n")
        

def eval_result():
    f1 = codecs.open('./eval.csv','r','gb18030')
    f2 = codecs.open('./submission_eval.csv','r','gb18030')
    count = 0
    precision = 0
    while 1:
        line1 = f1.readline()
        line2 = f2.readline()
        if not line1:
            break
        ground_truth = line1.split()[:3]
        predict_label = line2.split()[1:]
        if ground_truth[0] == predict_label[0]:
            precision += 1
        if ground_truth[1] == predict_label[1]:
            precision += 1
        if ground_truth[2] == predict_label[2]:
            precision += 1
        
        count += 1
    precision = precision / (3*count)
    print precision




if __name__ == '__main__':
    for line in codecs.open('whole_train.csv','r','gb18030'):
        read_rec(line)
    rerange()
    predict_label('./test.csv','./submission.csv','./submission_tmp.csv')
    #for line in codecs.open('train.csv','r','gb18030'):
    #    read_rec(line)
    #rerange()
    #predict_label('./eval.csv','./submission_eval.csv')
    #eval_result()




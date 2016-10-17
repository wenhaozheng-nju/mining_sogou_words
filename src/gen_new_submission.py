#!/usr/bin/env python
# coding=utf-8


pre_submission = open('./submission_pre.csv','r')
submission = open('./submission.csv','r')
new_submission = open('./submission_new.csv','w')

for line1 in pre_submission:
    line2 = submission.readline()
    eles1 = line1.split()
    eles2 = line2.split()[1:]
    eles2 = map(int,eles2)
    for i in range(len(eles2)):
        if eles2[i] != -1:
            eles1[i+1] = str(eles2[i])
    new_submission.write(" ".join(eles1)+"\n")




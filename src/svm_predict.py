#!/usr/bin/env python
# coding=utf-8

from sklearn.svm import SVC
import numpy as np
data = np.load('./user_tag_query.2W.TEST.docu_topic.npy')
clf = SVC()
clf.fit()

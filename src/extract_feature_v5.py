# TFIDF  LSIModel  LDAModel doc2vecmodel

import codecs
import numpy as np
import jieba.posseg as pseg
from gensim import corpora, models, matutils


class TaggedDocumentLines(object):
    def __init__(self, filenamelist):
        self.filenamelist = filenamelist
        self.count  = 0

    def __iter__(self):
        print self.count
        for filename in self.filenamelist:
            extag = filename.split('_')[1]
            print extag
            code = 'gb18030' if extag != 'copus' else 'utf8'
            for idx, line in enumerate(codecs.open(filename, 'r', code)):
                yield models.doc2vec.TaggedDocument(line.split(), [extag + '_%s' %idx])
        self.count += 1


def get_trainY(trainY_path):
    trainY = np.load(trainY_path)
    return trainY


def gen_doc2vec_model(text_file_list,save_path,vecdim,params=None):
    # generate doc2vec model from the data
    # sentences = []
    # f = codecs.open(text_file_list[0], 'r', 'gb18030')
    # for idx, line in enumerate(f.readlines()):
    #     sentences.append(models.doc2vec.TaggedDocument(line.split(), ['train_%s' %idx]))
    # f = codecs.open(text_file_list[1], 'r', 'gb18030')
    # for idx, line in enumerate(f.readlines()):
    #     sentences.append(models.doc2vec.TaggedDocument(line.split(), ['test_%s' %idx]))
    # print len(sentences)
    # set params
    window1 = 5
    min_count1 = 1
    worker1 = 4
    sample1 = 1e-4
    negative1 = 10
    iter1 = 20
    if params:
        if 'window' in params.keys():
            window1 = params['window']
        if 'min_count' in params.keys():
            min_count1 = params['min_count']
        if 'worker' in params.keys():
            worker1 = params['worker']
        if 'sample' in params.keys():
            sample1 = params['sample']
        if 'negative' in params.keys():
            negative1 = params['negative']
        if 'iter' in params.keys():
            iter1 = params['iter']

    doc2vec_model = models.Doc2Vec(TaggedDocumentLines(text_file_list), size=vecdim, window=window1, min_count=min_count1,
                                   workers=worker1, sample=sample1, negative=negative1, iter=iter1)
    doc2vec_model.save(save_path)
    return doc2vec_model


def get_doc2vec_model(model_path):
    doc2vec_model = models.Doc2Vec.load(model_path)
    return doc2vec_model


def gen_doc2vec_dataset(trainY ,doc2vecmodel, vecdim, data_path_list):
    trainX = np.zeros((20000, vecdim))
    for idx  in range(20000):
        trainX[idx] = doc2vecmodel.docvecs['train_%s' %idx]
    testX = np.zeros((20000, vecdim))
    for idx  in range(20000):
        testX[idx] = doc2vecmodel.docvecs['test_%s' %idx]
    traindataset = np.hstack((trainY, trainX))
    testdataset = testX
    print traindataset.shape
    print testdataset.shape
    np.save(data_path_list[0], traindataset)
    np.save(data_path_list[1], testdataset)
    return traindataset, testdataset


if __name__ == '__main__':
    # set the path
    trainnum = 20000
    testnum = 20000
    fold_path = './data/backup/'
    trainY_path = fold_path+'trainY.npy'
    text_path_tr = fold_path+ 'text_train'
    text_path_te = fold_path+ 'text_test'

    text_path_list = [text_path_tr,text_path_te]

    # tfidf_model_path = fold_path + 'filter_tfidfmodel'

    model_fold_path = fold_path + 'docvec1016/'
    doc2vec_model_path = model_fold_path + 'keyword_all_docvecmodel'

    train_data_path = model_fold_path+'docvec_feature_train'
    test_data_path = model_fold_path+ 'docvec_feature_test'
    data_path_list = [train_data_path,test_data_path]


    # gen_dict(text_path_list,dict_path)
    # gen_corpus(text_path_list,dict_path,corpus_path)
    # gen_tfidf_model(corpus_path,tfidf_model_path)

    topicnum = 100
    param = {'window':5, 'min_count':1, 'sample':1e-5, 'negative':10, 'iter':20}
    gen_doc2vec_model(text_path_list, doc2vec_model_path, topicnum, param)
    trainY = get_trainY(trainY_path)
    doc2vec_model = get_doc2vec_model(doc2vec_model_path)
    gen_doc2vec_dataset(trainY,doc2vec_model, topicnum, data_path_list)

    # tfidf_model = get_tftidf_model(tfidf_model_path)
    # gen_tfidf_dataset(trainY,traincorpus,testcorpus,tfidf_model,data_path_list)

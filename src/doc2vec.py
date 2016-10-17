
import codecs
from gensim import corpora, models, matutils
import numpy as np

my_dir = "../result/"

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

def get_doc2vec_model(model_path):
    doc2vec_model = models.Doc2Vec.load(model_path)
    return doc2vec_model

def get_trainY(trainY_path):
    trainY = np.load(trainY_path)
    return trainY

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


my_model = models.Word2Vec.load(my_dir+'word2vec.model')
my_model.save_word2vec_format(my_dir + 'vectors.txt',binary=False)

sources = (my_dir+'text_train',my_dir+'text_test')

docs = TaggedDocumentLines(sources)
model = models.Doc2Vec(dm=0,size=100,iter=20,min_count=2,workers=3,dbow_words=1,sample=1e-5,negative=10)

model.build_vocab(docs)
model.intersect_word2vec_format(my_dir+'vectors.txt',binary=False)
model.train(docs)
model.save(my_dir+'doc2vec.model')

train_data_path = my_dir+'docvec_feature_train'
test_data_path = my_dir+'docvec_feature_test'
data_path_list = [train_data_path,test_data_path]
topicnum = 100
trainY = get_trainY(my_dir + 'trainY.npy')
doc2vec_model = get_doc2vec_model(my_dir + 'doc2vec.model')
gen_doc2vec_dataset(trainY,doc2vec_model, topicnum, data_path_list)


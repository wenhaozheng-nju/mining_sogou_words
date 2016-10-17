import numpy as np
import xgboost as xgb
import codecs
from sklearn.svm import SVC
from scipy.optimize import fmin_powell


def loaddata():
    traindata = np.load('data//user_tag_query.2W.TRAIN.docu_topic.npy')
    testdata = np.load('data//user_tag_query.2W.TEST.docu_topic.npy')
    trainY = traindata[:,0:3]
    trainX = traindata[:,3:]
    testX = testdata
    return trainX, trainY, testX


def kfoldcv(trainX, trainY, k=10):
    foldnum = len(trainX)/k
    acclist1 = []
    acclist2 = []
    acclist3 = []
    print np.sum(trainY[:,0] == 0)
    print np.sum(trainY[:,1] == 0)
    print np.sum(trainY[:,2] == 0)
    for i in range(k):
        mask = range(i*foldnum, (i+1)*foldnum)
        cvevalY = trainY[mask]
        cvevalX = trainX[mask]
        mask = range(0, i*foldnum) + range((i+1)*foldnum, 20000)
        cvtrainY = trainY[mask]
        cvtrainX = trainX[mask]

        mask1 = cvtrainY[:,0]!=0
        dtrain1 = xgb.DMatrix(cvtrainX[mask1], label=cvtrainY[:,0][mask1]-1)
        mask1 = cvevalY[:,0]!=0
        deval1 = xgb.DMatrix(cvevalX[mask1])
        param = {'bst:max_depth':50, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax', 'nthread':4, 'num_class':6, 'min_child_weight':200}
        num_round = 60
        bst = xgb.train(param, dtrain1, num_round)
        # get prediction
        pred1 = bst.predict(deval1)
        acc1 = np.mean(pred1+1 == cvevalY[:,0][mask1])
        print acc1
        acclist1.append(acc1)

        mask2 = cvtrainY[:,1]!=0
        dtrain2 = xgb.DMatrix(cvtrainX[mask2], label=cvtrainY[:,1][mask2]-1)
        mask2 = cvevalY[:,1]!=0
        deval2 = xgb.DMatrix(cvevalX[mask2])
        param = {'bst:max_depth':50, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax', 'nthread':4, 'num_class':2, 'min_child_weight':300}
        num_round = 60
        bst = xgb.train(param, dtrain2, num_round)
        # get prediction
        pred2 = bst.predict(deval2)
        acc2 = np.mean(pred2+1 == cvevalY[:,1][mask2])

        # mask2 = cvtrainY[:,1]!=0
        # cvtrainX2 = cvtrainX[mask2]
        # cvtrainY2 = cvtrainY[:,1][mask2]
        # mask2 = cvevalY[:,1]!=0
        # cvevalX2 = cvevalX[mask2]
        # cvevalY2 = cvevalY[:,1][mask2]
        # clf = SVC(kernel='linear', C=0.5)
        # clf.fit(cvtrainX2, cvtrainY2)
        # pred2 = clf.predict(cvevalX2)
        # print pred2
        # print [sum(pred2 == i) for i in range(1, 3)]
        # print [sum(cvevalY2 == i) for i in range(1, 3)]
        # acc2 = np.mean(pred2 == cvevalY2)

        print acc2
        acclist2.append(acc2)

        mask3 = cvtrainY[:,2]!=0
        dtrain3 = xgb.DMatrix(cvtrainX[mask3], label=cvtrainY[:,2][mask3]-1)
        mask3 = cvevalY[:,2]!=0
        deval3 = xgb.DMatrix(cvevalX[mask3])
        param = {'bst:max_depth':50, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax', 'nthread':4, 'num_class':6, 'min_child_weight':150}
        num_round = 60
        bst = xgb.train(param, dtrain3, num_round)
        # get prediction
        pred3 = bst.predict(deval3)
        acc3 = np.mean(pred3+1 == cvevalY[:,2][mask3])
        print acc3
        acclist3.append(acc3)

    print np.mean(acclist1)
    print np.mean(acclist2)
    print np.mean(acclist3)

def eval_wrapper(yhat,y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat),np.min(y),np.max(y)).astype(int)
    return np.sum((yhat-y) == 0)

def apply_offset(data,bin_offset,sv,scorer=eval_wrapper):
    data[1,data[0].astype(int)==sv] = data[0,data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1],data[2])
    return score


def offset(train_preds,test_preds,train_groudtruth,num_classes):
    offsets = np.ones(num_classes) * -0.5
    offset_train_preds = np.vstack((train_preds,train_preds,train_groudtruth))
    for j in range(num_classes):
        j = j+1
        train_offset = lambda x: - apply_offset(offset_train_preds,x,j)
        offsets[j] = fmin_powell(train_offset,offsets[j])
    test_groundtruth = np.array([0]*len(test_preds))
    data = np.vstack((test_preds,test_preds,test_groundtruth))
    for j in range(num_classes):
        j = j+1
        data[1,data[0].astype(int)==j] = data[0,data[0].astype(int)==j] + offsets[j]
    final_test_pred = np.round(np.clip(data[1],1,num_classes)).astype(int)
    return final_test_pred



if __name__ == '__main__':
    trainX, trainY, testX = loaddata()
    # kfoldcv(trainX, trainY, 10)

    testY = np.zeros((len(testX), 3))
    dtest = xgb.DMatrix(testX)

    mask1 = trainY[:,0]!=0
    dtrain1 = xgb.DMatrix(trainX[mask1], label=trainY[:,0][mask1]-1)
    param = {'bst:max_depth':50, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax', 'nthread':4, 'num_class':6, 'min_child_weight':200}
    num_round = 60
    bst1 = xgb.train(param, dtrain1, num_round)
    # get prediction
    testY[:,0] = bst1.predict(dtest)+1

    mask2 = trainY[:,1]!=0
    dtrain2 = xgb.DMatrix(trainX[mask2], label=trainY[:,1][mask2]-1)
    param = {'bst:max_depth':50, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax', 'nthread':4, 'num_class':2, 'min_child_weight':300}
    num_round = 60
    bst2 = xgb.train(param, dtrain2, num_round)
    # get prediction
    testY[:,1] = bst2.predict(dtest)+1

    mask3 = trainY[:,2]!=0
    dtrain3 = xgb.DMatrix(trainX[mask3], label=trainY[:,2][mask3]-1)
    param = {'bst:max_depth':50, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax', 'nthread':4, 'num_class':6, 'min_child_weight':150}
    num_round = 60
    bst3 = xgb.train(param, dtrain3, num_round)
    # get prediction
    testY[:,2] = bst3.predict(dtest)+1

    print testY
    print [sum(testY[:,0] == i)for i in range(1, 7)]
    print [sum(testY[:,1] == i) for i in range(1, 3)]
    print [sum(testY[:,2] == i) for i in range(1, 7)]

    f = codecs.open('data//user_tag_query.2W.TEST.keyword', 'r', 'gbk18030')
    lines = []
    for idx, line in enumerate(f.readlines()):
        seg = line.split()
        lines.append(seg[0] + ' ' +  str(testY[idx, 0]) + ' ' +  str(testY[idx, 1]) + ' ' +  str(testY[idx, 2]) + '\n')
    f.close()
    print lines
    f = codecs.open('data//prediction.csv', 'w', 'gbk18030')
    f.writelines(lines)
    f.close()







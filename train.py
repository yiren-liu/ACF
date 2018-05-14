__author__ = 'jingyuan'
from New_Dataset import *
from Model import *
from Test_Dataset import *
import ImpDataset
from time import time
import cPickle
import os
dataset = "Vine"

train_file = "Input/train.tsv"
test_file = "Input/test.tsv"

splitter = "\t"
hold_k_out = 1
batch_size = 128

num_epoch = 400
dim = 128
trainset=[]
if os.path.exists('Input/trainset.save'):
    f=file('Input/trainset.save', 'rb')
    trainset=cPickle.load(f)
    f.close()
    print('Load trainset from save')
else:
    trainset = New_Dataset(train_file, splitter, batch_size)
    f = file('Input/trainset.save', 'wb')
    cPickle.dump(trainset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
print('Load trainset sucessfully')
testset = Test_Dataset(test_file, splitter,batch_size,trainset)
print('Load testset sucessfully')
testDataset2=ImpDataset.ImpDataset('')
print('Load testDataset2 sucessfully')
lr = 0.0001
reg = 0.01
topK=10



model = Model(trainset,testset,testDataset2,trainset.num_user, trainset.num_item, dim, reg, lr, 'Model/')
print('Creat model sucessfully')

# model.train(trainset.epoch)  #for test

t1 = time()
ahit, andcg=model.test(topK)
best_hr, best_ndcg, best_iter = ahit, andcg, -1
print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (ahit, andcg, time()-t1))


for i in xrange(1,num_epoch):
    model.train(trainset.epoch)
    t2 = time()
    ahit, andcg = model.test(topK,i)
    print('Epoch %s: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (i, ahit, andcg, t2 - t1))
    if ahit > best_hr:
        best_hr = ahit
        best_iter = i
        model.save('Model/')
    if andcg > best_ndcg:
        best_ndcg = andcg



print 'bestmodel saved!'

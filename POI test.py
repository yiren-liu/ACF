from Dataset import Dataset
from Model import *
from Test_Dataset import *
splitter = "\t"
train_file = "train.tsv"
test_file="test.tsv"

batch_size = 10
dim = 128
dataset = "Vine"



trainset = Dataset(train_file, splitter,10, batch_size,dim = 128)
users_b, pos_items_b, neg_items_b,new_items_u,mask,items_feature=trainset.get_batch()
testset = Test_Dataset(test_file, splitter,batch_size,trainset)



lr = 0.01
reg = 0.01
model = Model(trainset,trainset,trainset.num_user, trainset.num_item, dim, reg, lr, None)
model.train(trainset.epoch)
model.save('./model/'+dataset+'/0_')
__author__ = 'jingyuan'
import numpy as np
from sets import Set
import h5py
import ast
from random import randint
from copy import deepcopy
class Test_Dataset(object):
    def __init__(self, filename, splitter, batch_size, trainset): #line formart: [usr]\t[i]\t[j]\t
        self.trainset = trainset
        self.maxbatch = batch_size

        lines = map(lambda x: x.strip().split(splitter), open(filename,'r').readlines())
        self.usr = map(lambda line: line[0], lines)
        self.v_i = map(lambda line: line[1], lines)
        self.usrTimePhoto = map(lambda line: (line[0], int(line[3]), line[4]), lines)

        self.epoch = len(self.usr) / self.maxbatch
        if len(self.usr) % self.maxbatch != 0:
            self.epoch += 1

        # tmp = zip(self.usr,self.v_i,self.usrTimePhoto)
        # tmp.sort(lambda x, y: int(y[2])-int(x[2]))
        # self.usr, self.v_i, self.usrTimePhoto = zip(*tmp)

    def get_u_list(self,user_list):
        return self.trainset.u_list_map[user_list]

    def gen_batch(self,datas):
        dd = deepcopy(datas)
        max_item_count = len(dd[0])
        for uu in dd:
            for i in xrange(max_item_count-len(uu)):
                uu.append(-1)
        return dd

    def genitemmask(self,itemnum):

        maxnum = itemnum[0]
        mask = np.asarray(map(lambda num:[1]*num + [0]*(maxnum-num),itemnum),dtype='int32')
        return mask

    def get_batch(self,i):
        user_list = self.usr[i*self.maxbatch:(i+1)*self.maxbatch]
        users = np.asarray(user_list,dtype=np.int32)

        pos_items = np.asarray(self.v_i[i * self.maxbatch:(i + 1) * self.maxbatch], dtype=np.int32)

        userTimePhotoList = self.usrTimePhoto[i * self.maxbatch:(i + 1) * self.maxbatch]
        temp = self.trainset.get_u_list(userTimePhotoList)
        temp_u_list, mask_picture = self.trainset.gen_batch(temp)
        feats_idx = np.array(temp_u_list).astype(np.int32)


        mask_time = self.trainset.GetBatchMaskTime(userTimePhotoList)

        return users,mask_picture,mask_time,feats_idx, pos_items




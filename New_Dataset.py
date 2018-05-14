__author__ = 'jingyuan'
import numpy as np
from sets import Set
import h5py
import ast
from random import randint
import cPickle
import random
from copy import deepcopy
import pandas as pd


class New_Dataset(object):
    def __init__(self, filename, splitter, batch_size): #line formart: [usr]\t[i]\t[j]\t[list]\t
        self.maxbatch = batch_size
        # lines=[]
        # with open(filename,'r') as testFile:
        #     for i in range(1000):
        #         lines.append(testFile.readline().strip().split(splitter))
        lines = map(lambda x: x.strip().split(splitter), open(filename).readlines())
        self.usr = map(lambda line: line[0], lines)
        self.v_i = map(lambda line: line[1], lines)
        self.usrTimePhoto = map(lambda line: (line[0],int(line[3]),line[4]), lines)


        self.num_user = len(set(self.usr))
        self.num_item = len(set(self.v_i))

        print 'num_user ',self.num_user
        print 'num_item ',self.num_item

        self.epoch = len(self.usr) / self.maxbatch
        if len(self.usr) % self.maxbatch != 0:
            self.epoch += 1

        self.videoIndex, self.video_features = self.load_frame_feat()
        #self.videoIndex.to_csv('Output/videoIndex.csv',index=False,sep=',')
        self.postiveOfUser=self.GenPostive()
        self.u_list_map,self.mask_time = self.load_u_list_map()

    def GenPostive(self):

        userVenueDict = dict()
        for x in range(len(self.usr)):
            if self.usr[x] not in userVenueDict:
                userVenueDict[self.usr[x]] = set()
            userVenueDict[self.usr[x]].add(self.v_i[x])
        return userVenueDict

    def shuffle_data(self):
        c = list(zip(self.usr, self.v_i, self.usrTimePhoto))
        random.shuffle(c)
        self.usr, self.v_i, self.usrTimePhoto = zip(*c)

    def load_frame_feat(self):
#        frame_feat = cPickle.load(open('/home/jie/jingyuan/sigir/data/frame_feat.p','rb'))
        photoid = []
        featureData = []
        with open('Input/name-fc8.txt', 'r') as file:
            for line in file.readlines():
                a,b=line.split(' ', 1)
                a=eval(a)[0].split('.')[0]
                b=eval(b)
                if a in photoid:
                    continue
                photoid.append(a)
                featureData.append(b)
        #with open('Output/photoid.csv','w') as f:
            #f.write(str(photoid))
            #f.close()
        photoid=pd.factorize(pd.Series(photoid))[1]
        featureData=np.array(featureData,dtype='float32')
        return photoid, featureData


    def load_u_list_map(self):
#        u_list_map = cPickle.load(open('/home/jie/jingyuan/sigir/data/user_all_vines.p','rb'))
        with open('Input/dict1.txt') as f1:
            dict1=eval(f1.readline())
        with open('Input/dict2.txt') as f2:
            dict2=eval(f2.readline())
        return dict1,dict2

    def gen_batch(self,datas):

        #max_item_count = len(dd[0])
        item_counts=[]
        for xx in datas:
            for m in xx:
                item_counts.append(len(m))
        max_item_count = max(item_counts)

        photoIndice=[]
        mask=[]
        for ixx,xx in enumerate(datas):
            featTemp1=[]
            maskTemp1=[]
            for im,m in enumerate(xx):
                featTemp2=[]
                maskTemp2 = []
                for i in xrange(max_item_count):
                    if i < len(m):
                        try:
                            featTemp2.append(self.videoIndex.get_loc(datas[ixx][im][i][2]))
                            maskTemp2.append(1)
                        except Exception as e:
                            # print 'Error, There is not this photo in picture features file'
                            featTemp2.append(-1)
                            maskTemp2.append(0)

                    else:
                        featTemp2.append(-1)
                        maskTemp2.append(0)
                featTemp1.append(featTemp2)
                maskTemp1.append(maskTemp2)
            photoIndice.append(featTemp1)
            mask.append(maskTemp1)

        return photoIndice,mask

    def genitemmask(self,itemnum):

        maxnum = max(xx for xx in itemnum)
        mask = np.asarray(map(lambda num:[1]*num + [0]*(maxnum-num),itemnum),dtype='int32')
        return mask

    def genvideofeature(self,temp_u_list):
        return self.video_features[temp_u_list]

    def genneg(self,user_list):
        neg_items = []
        for i in xrange(len(user_list)):
            i_id = randint(0,self.num_item-1)
            while i_id in self.postiveOfUser[str(i)]:
                i_id = randint(0,self.num_item-1)
            neg_items.append(i_id)
        return neg_items

    def get_u_list(self,user_list):
        batchUserPhoto=[]
        for user in user_list:
            temp=[]
            for i in xrange(1,11):
                temp.append(self.u_list_map[user][i])
            batchUserPhoto.append(temp)

        return batchUserPhoto

    def GetBatchMaskTime(self,batch):
        mask=[]
        for sample in batch:
            m=self.mask_time[sample]
            mask.append(m)

        return mask

    def get_batch(self,i):

        user_list = self.usr[i*self.maxbatch:(i+1)*self.maxbatch]
        users = np.asarray(user_list,dtype=np.int32)

        pos_items = np.asarray(self.v_i[i*self.maxbatch:(i+1)*self.maxbatch],dtype=np.int32)
        neg_items = np.asarray(self.genneg(user_list),dtype=np.int32)

        userTimePhotoList = self.usrTimePhoto[i * self.maxbatch:(i + 1) * self.maxbatch]
        temp = self.get_u_list(userTimePhotoList) #get batchsize*10*m*3 and 3 is (user,time,photo_id) per user get items of users before the train time
        temp_u_list,mask_picture = self.gen_batch(temp) #[user,item] after append -1
        feats_idx = np.array(temp_u_list).astype(np.int32)  # object to int

        mask_time=self.GetBatchMaskTime(userTimePhotoList)

        return (users, pos_items, neg_items,mask_picture,mask_time,feats_idx)

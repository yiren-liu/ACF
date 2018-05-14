__author__ = 'jingyuan'

import theano
import theano.tensor as T
from UsrEmblayer import *
from VidEmblayer import *
from GetuEmbLayer import *
from GetvEmbLayer import *
from AttentionLayer_Feat import *
from AttentionLayer_Item import *
from ContentEmbLayer import *
import numpy as np
import math
import evaluation

def save_attent(time_attent, img_attent):
    with open(r'Output\attent.txt','w') as f:
        f.write(str(time_attent) +'\n')
        f.write(str(img_attent) + '\n')
        print('sucessfully written weight!')

def lower_dim(array):
    temp = []
    for i in array:
        try:
            temp=np.concatenate([temp,i])
        except Exception,e:
            print 'lower_dim_err:',e
            temp=i
    return temp


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def softmask(x):
    y = np.exp(x)
    #y = y * mask
    sumx = np.sum(y,axis=0)
    #x = y/sumx.dimshuffle(0,'x')
    x=y/sumx
    return x

class Model(object):
    def __init__(self,trainset,testset,testDataset2,num_user,num_item,dim,reg,lr,prefix):
        self.trainset = trainset
        self.testset = testset
        self.testDataset2=testDataset2
        self.reg = numpy.float32(reg)
        self.lr = numpy.float32(lr)
        self.num_item = num_item
        self.video_features = self.trainset.video_features

        T.config.compute_test_value = 'warn'

        u = T.ivector('u') #[num_sample,]
        iv = T.ivector('iv') #[num_sample,]
        jv = T.ivector('jv') #[num_sample,]
        mask_frame = T.itensor3('mask_frame')  #[num_sample, num_video, num_frame]
        mask = T.imatrix('mask') #[num_sample, num_video]

        feat=T.ftensor4('feat')

        u.tag.test_value = np.asarray([0,1,2],dtype='int32')
        iv.tag.test_value = np.asarray([4,5,2],dtype='int32')
        jv.tag.test_value = np.asarray([1,3,0],dtype='int32')
        mask.tag.test_value = np.asarray([[1,1,0],[1,0,0],[1,1,1]],dtype='int32')
        # feat_idx.tag.test_value = np.asarray([[3,4,-1],[5,-1,-1],[6,2,4]],dtype='int32')


        rng = np.random
        layers = []

        Uemb = UsrEmblayer(rng,num_user,dim,'usremblayer',prefix)
        Vemb = VidEmblayer(rng,num_item,dim,'videmblayer',prefix)

        layers.append(Uemb)
        layers.append(Vemb)
        uemb_vec = GetuEmbLayer(u,Uemb.output,'uemb',prefix)
        iemb_vec = GetvEmbLayer(iv,Vemb.output,'v1emb',prefix)
        jemb_vec = GetvEmbLayer(jv,Vemb.output,'v2emb',prefix)

        layers.append(AttentionLayer_Feat(rng, 1000, uemb_vec.output, feat, dim, dim, mask_frame, 'attentionlayer_feat',prefix))

        layers.append(AttentionLayer_Item(rng, uemb_vec.output, layers[-1].output,dim,dim,mask,'attentionlayer_item',prefix))

        u_vec = uemb_vec.output + layers[-1].output
        self.layers = layers
        y_ui = T.dot(u_vec, iemb_vec.output.T).diagonal()
        y_uj = T.dot(u_vec, jemb_vec.output.T).diagonal()
        self.params = []
        loss = - T.sum(T.log(T.nnet.sigmoid(y_ui - y_uj)))
        for layer in layers:
            self.params += layer.params #[U,V,W_Tran,Wu,Wv,b,c]
        #regularizer = self.reg * ((uemb_vec.output ** 2).sum() + (iemb_vec.output ** 2).sum() + (jemb_vec.output ** 2).sum() +
        #                          (self.params[2] ** 2).sum() + (self.params[3] ** 2).sum() + (self.params[4] ** 2).sum() +
        #                            (self.params[5] ** 2).sum())

        regularizer = self.reg * ((uemb_vec.output ** 2).sum() + (iemb_vec.output ** 2).sum() + (jemb_vec.output ** 2).sum() )

        for param in self.params[2:]:
            regularizer += self.reg * (param ** 2).sum()

        loss = regularizer + loss

        updates = [(param, param-self.lr*T.grad(loss,param)) for param in self.params]

        self.train_model = theano.function(
            inputs = [u,iv,jv,mask_frame,mask,feat],
            outputs = loss,
            updates=updates
        )

        self.test_model = theano.function(
            inputs = [u,mask_frame,mask,feat],
            outputs= [u_vec,Vemb.output,layers[-1].atten,layers[-2].atten],#for test: layers[-2].output,layers[-2].items_emb,layers[-2].atten
        )

    def train(self, iters):
        self.trainset.shuffle_data()
        lst = np.random.randint(self.trainset.epoch, size=iters)
        n = 0
        for i in lst:
            n += 1
            users, pos_items, neg_items, mask_frame, mask, feat_idx = self.trainset.get_batch(i)
            feat=self.video_features.take(feat_idx,axis=0)
            out= self.train_model(users,pos_items,neg_items,mask_frame,mask,feat)
            # if n%100==0:
            print n, 'cost:', out

    def test(self,topK,epoch_num=0):
        total_ahit, total_andcg=[],[]
        for i in xrange(self.testset.epoch):#
            # user_list,pos_items, neg_items, mask_frame, mask, feats_idx= self.trainset.get_batch(i)
            user_list, mask_frame,mask,feats_idx,pos_items = self.testset.get_batch(i)
            feat = self.video_features.take(feats_idx, axis=0)
            [user_vector, V_matrix, time_attent, img_attent] = self.test_model(user_list, mask_frame,mask,feat)
            #V_value = np.asarray(V_matrix.eval())
            score_maxtrix = np.dot(user_vector,V_matrix.T)
            # index_top_K = score_maxtrix.argsort()[:,-topK:][:,::-1]

            ahit, andcg=evaluation.evaluate_model(score_maxtrix,self.testDataset2,topK,user_list,pos_items)
            total_ahit.append(ahit)
            total_andcg.append(andcg)
            #save_attent(time_attent, img_attent, i)
            print(i)
            #print type(time_attent)
            #print type(img_attent)
            #print img_attent.shape
            img_attent = lower_dim(img_attent)
            feats_idx = lower_dim(feats_idx)
            try:
                time_attent0 = np.loadtxt("Output/epoch_"+str(epoch_num)+"_time_attent.csv", delimiter=",")
                user_list0 = np.loadtxt("Output/epoch_"+str(epoch_num)+"_user_list.csv", delimiter=",")
                img_attent0 = np.loadtxt("Output/epoch_" + str(epoch_num) + "_img_attent.csv", delimiter=",")
                feats_idx0 = np.loadtxt("Output/epoch_" + str(epoch_num) + "_feats_idx.csv", delimiter=",")
                #print time_attent0
                time_attent = np.concatenate([time_attent0, time_attent])
                user_list = np.concatenate([user_list0, user_list])
                img_attent = np.concatenate([img_attent0, img_attent])
                feats_idx = np.concatenate([feats_idx0, feats_idx])
            except Exception,e:
                #print 'time_attent.csv does note exist, creating'
                print Exception, ":", e
            np.savetxt("Output/epoch_"+str(epoch_num)+"_time_attent.csv", time_attent, delimiter=",")
            np.savetxt("Output/epoch_"+str(epoch_num)+"_user_list.csv", user_list, delimiter=",")
            np.savetxt("Output/epoch_" + str(epoch_num) + "_img_attent.csv", img_attent, delimiter=",")
            np.savetxt("Output/epoch_" + str(epoch_num) + "_feats_idx.csv", feats_idx, delimiter=",")
            #np.savetxt(r"Output\img_attent.csv", img_attent, delimiter=",")

        return np.asarray(total_ahit).mean(), np.asarray(total_andcg).mean()


    def save(self, prefix):
        for layer in self.layers:
            layer.save(prefix)



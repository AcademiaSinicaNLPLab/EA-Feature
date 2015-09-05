# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import numpy as np
import cPickle as pickle
'''
先把文章切成各個context part
然後把每個context part的各個sentence的w2v feature vectors加起來除sentence數，作為這個context part的feature vector
然後concatenate每個part的feature vector成為文章的feature vector

'''
def help():
    print "usage: python [featureNpz_save_path][number_of_parts]"
    print
    print " e.g.: python  /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+NewContextInfo2_Maxis_series.Xy.npz 2"
    exit(-1)


if __name__ == '__main__':

    if len(sys.argv) != 3: help()

    number_of_parts = int(sys.argv[2])
    count = 0

    X = np.zeros((40000,300*number_of_parts),dtype="float32")
    y = np.array([])

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    # emotions = emotions[0:1]
    # print emotions
    for i,emotion in enumerate(emotions):

        print 'Start to load w2vStanford_rmP_sentences.'+emotion+'.Xy.pkl'
        docs = pickle.load( open('/corpus/LJ40K/data/featurevecs/raw/w2vStanford_rmP_sentences/w2vStanford_rmP_sentences.'+emotion+'.Xy.pkl', "rb" ) )
        
        for i in xrange(len(docs)):

            SentenceVecs = docs[i]['X']
            n_sentences = SentenceVecs.shape[0]
            n_features = SentenceVecs.shape[1]

            DocVec = np.array([])
            # print 'SentenceVecs :　', SentenceVecs.shape
            if n_sentences < number_of_parts:
                first_sentence = SentenceVecs[0,:]
                other_sentences = np.reshape(SentenceVecs[1:,:], np.product(SentenceVecs[1:,:].shape))
                adding_sentences = np.tile(first_sentence, number_of_parts-n_sentences)
                DocVec = np.concatenate((first_sentence, adding_sentences, other_sentences),axis=1)
            else:
                avg_number_of_sentences = n_sentences/ number_of_parts
                number_of_sentences_remain = n_sentences % number_of_parts
                start_row = 0
                for i in xrange(number_of_parts):
                    PartVec = np.zeros((n_features,),dtype="float32")
                    if number_of_sentences_remain <= 0:
                        Part_of_SentenceVecs = SentenceVecs[start_row:start_row+avg_number_of_sentences,:]
                    else:
                        Part_of_SentenceVecs = SentenceVecs[start_row:start_row+avg_number_of_sentences+1,:]
                    start_row = start_row+Part_of_SentenceVecs.shape[0]
                    SentenceVec_sum = np.zeros((n_features,),dtype="float32")
                    for row in xrange(Part_of_SentenceVecs.shape[0]):
                        SentenceVec_sum = np.add(SentenceVec_sum,Part_of_SentenceVecs[row])
                    PartVec = np.divide(SentenceVec_sum,Part_of_SentenceVecs.shape[0])
                    DocVec = np.concatenate((DocVec, PartVec),axis=1)
                    number_of_sentences_remain = number_of_sentences_remain - 1

            X[count] = DocVec
            print count
            count = count + 1

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)
        print ">> %s finished! " % emotion
        del docs
    np.savez_compressed(sys.argv[1], X=X, y=y)

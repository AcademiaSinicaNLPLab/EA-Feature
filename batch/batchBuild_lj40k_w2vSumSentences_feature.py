# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import numpy as np
import cPickle as pickle
'''
把各個sentence的w2v feature vectors加起來後除sentence數，作為這個文章的feature vector

'''
def help():
    print "usage: python [featureNpz_save_path][number_of_parts]"
    print
    print " e.g.: python  /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+SumSentences_Maxis_series.Xy.npz"
    exit(-1)


if __name__ == '__main__':

    if len(sys.argv) != 2: help()

    count = 0

    X = np.zeros((40000,300),dtype="float32")
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
            DocVec = np.zeros((n_features,),dtype="float32")

            # print 'SentenceVecs :　', SentenceVecs.shape
            for row in xrange(n_sentences):
                DocVec = np.add(DocVec,SentenceVecs[row])
            DocVec = np.divide(DocVec,n_sentences)

            X[count] = DocVec
            print count
            count = count + 1

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)
        print ">> %s finished! " % emotion
        del docs
    np.savez_compressed(sys.argv[1], X=X, y=y)
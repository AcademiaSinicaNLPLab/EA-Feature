# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from core import BuildFeature, Preprocessing
from sklearn.feature_extraction import DictVectorizer


def help():
    print "usage: python [tf type][idf type]"
    print
    print " e.g.: python tf3 idf2"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 3: help()
    y1 = np.array([])
    y2 = np.array([])

    print 'load words'
    Words_info = pickle.load(open('/corpus/LJ40K/data/features/tfidf/Words_info_0_800_tf1_tf2_tf3k1b03_idf1_idf2_idf3.pkl','rb'))
    print 'load docs'
    Docs_info = pickle.load(open('/corpus/LJ40K/data/features/tfidf/Docs_info_0_800_tf1_tf2_tf3k1b03_idf1_idf2_idf3.pkl','rb'))
    print 'load test docs'
    Docs_info_testdata = pickle.load(open('/corpus/LJ40K/data/features/tfidf/Docs_info_800_1000_tf1_tf2_tf3k1b03.pkl','rb'))

    tf = sys.argv[1]
    idf = sys.argv[2]

    print 'make word set'
    words_set = set()
    for word in Words_info:
        if Words_info[word]['word_total_count'] >= 10:
            words_set.add(word)

    print len(words_set)
    print 'use word set to build train tfidf'
    docs_tfidf = []
    for emotion in Docs_info:
        print emotion
        for doc in Docs_info[emotion]:
            doc_tfidf = {}
            for word in Docs_info[emotion][doc][tf+idf]:
                if word in words_set:
                    doc_tfidf[word] = Docs_info[emotion][doc][tf+idf][word]

            docs_tfidf.append(doc_tfidf)

        label = np.array([emotion]*len(Docs_info[emotion]))
        y1 = np.concatenate((y1, label), axis=1)


    v = DictVectorizer(sparse=False)
    print 'train build feature vecs'
    X1 = v.fit_transform(docs_tfidf)
    print X1.shape
    # print 'train dump'
    # np.savez_compressed('/home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/'+tf+'k1b03'+idf+'.Xy.train.npz', X=X1, y=y1)
    print 'training data ok!'


    print 'use word set to build test tfidf'
    docs_test_tfidf = []
    for emotion in Docs_info_testdata:
        print emotion
        for doc in Docs_info_testdata[emotion]:
            doc_tfidf = {}
            for word in Docs_info_testdata[emotion][doc][tf]:
                if word in words_set:
                    doc_tfidf[word] = Docs_info_testdata[emotion][doc][tf][word] * Words_info[word][idf]

            docs_test_tfidf.append(doc_tfidf)

        label = np.array([emotion]*len(Docs_info_testdata[emotion]))
        # label = np.array([emotion]*1000)
        y2 = np.concatenate((y2, label), axis=1)
    
    print 'test build feature vecs'
    X2 = v.transform(docs_test_tfidf)
    print X2.shape
    print 'test dump'
    print 'testing data ok!'

    np.savez_compressed('/home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/'+tf+'k1b03'+idf+'.Xy.test.npz', X=X2, y=y2)
    # print 'concatenate'
    # X3 = np.concatenate((X1, X2), axis=0)
    # np.savez_compressed('/home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/'+tf+'k1b03'+idf+'.Xy.npz', X=X3, y=y2)
    print 'finish!'


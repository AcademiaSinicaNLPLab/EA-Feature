# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from core import BuildFeature, Preprocessing
import pymongo
from pymongo import MongoClient
'''
把各個keyword的w2v feature vectors加起來後除取出的keyword數，作為這個文章的feature vector

'''
def help():
    print "usage: python [model_load_path][pkl_load_path][npz_save_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /corpus/LJ40K/data/pkl/lj40k_wordlists /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+SumKeywords_basic.Xy.npz"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 4: help()
    model = Word2Vec.load(sys.argv[1])
    n_features = model.syn0.shape[1]
    index2word_set = set(model.index2word)

    ## select mongo collections
    client = MongoClient('doraemon.iis.sinica.edu.tw:27017')
    LJ40K = client.LJ40K
    keyword_source = LJ40K['resource.WordNetAffect']
    # extend_keyword = list(keyword_source.find({'type':'extend'}))
    # basic_keyword = list(keyword_source.find({'type':'basic'}))
    extend_keyword_list = [ mdoc['word'].encode('utf-8') for mdoc in list(keyword_source.find({'type':'extend'}))]
    basic_keyword_list = [ mdoc['word'].encode('utf-8') for mdoc in list(keyword_source.find({'type':'basic'}))]

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    y = np.array([])

    load_path = sys.argv[2]
    save_path = sys.argv[3]

    new_docs = []
    empty_docs = set()
    # emotions = emotions[0:1]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_wordlists.pkl', "rb" ) )
        # docs = docs[0:5]
        for j,doc in enumerate(docs):
            new_doc_keyword = []
            new_doc = sum(doc,[])
            for word in new_doc:
                if word in basic_keyword_list and word in index2word_set:
                    new_doc_keyword.append(word)
            
            # print new_doc_keyword
            if len(new_doc_keyword) == 0:
                empty_docs.add((emotion,j))
                new_docs.append(new_doc)
            else:
                new_docs.append(new_doc_keyword)

            if (j+1)%100. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))   

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)
        print ">> %s. %s emotion finishing!" % (i+1,emotion)

    print empty_docs
    
    X = BuildFeature.getAvgFeatureVecs(new_docs, model)
    np.savez_compressed(save_path, X=X, y=y)


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
把各個含keyword的sentence的w2v feature vectors加起來後除取出的sentence數，作為這個文章的feature vector

'''
def help():
    print "usage: python [model_load_path][pkl_load_path][npz_save_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /corpus/LJ40K/data/pkl/lj40k_wordlists /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+SumKeywordSentences_basic_Maxis_series.Xy.npz"
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

    X = np.zeros((40000,300),dtype="float32")
    y = np.array([])

    load_path = sys.argv[2]
    save_path = sys.argv[3]

    idx = 0
    empty_docs_index = {}
    # emotions = emotions[0:1]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_wordlists.pkl', "rb" ) )
        empty_docs = set()
        for j,doc in enumerate(docs):
            new_doc_with_keywords = []
            new_doc = []

            for wordlist in doc:
                Use_this_wordlist = False
                keyword_inside = False
                wordlist = Preprocessing.Wordlist_cleaner(wordlist,remove_puncts=True)

                if len(wordlist) > 0 :
                    for word in wordlist:
                        if word in index2word_set:
                            Use_this_wordlist = True
                            break
                    for word in wordlist:
                        # if word in basic_keyword_list and word in index2word_set:
                        if word in extend_keyword_list and word in index2word_set:
                            keyword_inside = True
                            break
                    if Use_this_wordlist == True:
                        new_doc.append(wordlist)
                    if keyword_inside == True:
                        new_doc_with_keywords.append(wordlist)

            if len(new_doc) == 0:
                empty_docs.add(j)
                new_doc = [new_doc]

            if len(new_doc_with_keywords) == 0:
                SentenceVecs = BuildFeature.getAvgFeatureVecs(new_doc, model)
            else:
                SentenceVecs = BuildFeature.getAvgFeatureVecs(new_doc_with_keywords, model)
            
            n_sentences = SentenceVecs.shape[0]
            DocVec = np.zeros((n_features,),dtype="float32")

            # print 'SentenceVecs :　', SentenceVecs.shape
            for row in xrange(n_sentences):
                DocVec = np.add(DocVec,SentenceVecs[row])
            DocVec = np.divide(DocVec,n_sentences)
            del SentenceVecs

            X[idx] = DocVec
            idx = idx + 1

            if (j+1)%100. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))

            del new_doc
            del new_doc_with_keywords
            del DocVec

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)

        empty_docs_index[emotion] = empty_docs

        print ">> %s. %s emotion finishing!" % (i+1,emotion)
        # print emotion," ",empty_docs
        del empty_docs
        del docs

    np.savez_compressed(save_path, X=X, y=y)
    ## as the result of the experiment, no doc empty (means that all docs have useful sentences, the word of which can be found in w2v embeding)
    ## so below is not needed then
    # pickle.dump(empty_docs_index, open('empty_docs_index_sumkeywordsentences.pkl', 'wb'))

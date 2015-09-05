# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from core import BuildFeature, Preprocessing

'''
this batch output a npz file which contain feature information
inside npz file: 'X': array, 'y': array
'''
'''
把所有word w2v vectors 加起來除以word數量作為document的feature vector
跟batchBuild_lj2m_w2v_feature.py要做的目的相同，但是這版本產出 npz !!!!!!!!!!!!!!!!! 
而batchBuild_lj2m_w2v_feature.py則產出pkl(裡面是dictionary資料結構，最終還是numpy array)
想要產出什麼檔案，端看之後train的program的input需求

'''

def help():
    print "usage: python [model_load_path][wordlists_load_path][npz_save_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /corpus/LJ40K/data/pkl/lj40k_wordlists /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP_Maxis_series.Xy.npz"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 4: help()
    model = Word2Vec.load(sys.argv[1])
    index2word_set = set(model.index2word)

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    load_path = sys.argv[2]
    save_path = sys.argv[3]

    empty_docs_index = {}
    new_docs = []
    y = np.array([])

    # emotions = emotions[0:2]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_wordlists.pkl', "rb" ) )

        empty_docs = set()
        for j,doc in enumerate(docs):
            words = []
            # combine wordlists of one doc in order to get a list of words
            for wordlist in doc:
                words += wordlist
            words = Preprocessing.Wordlist_cleaner(words,remove_puncts=True)
            
            # words = ['i','am','good'] or words = []
            Use_this_words = False
            for word in words:
                if word in index2word_set:
                    Use_this_words = True
                    break
            if Use_this_words == False:
                empty_docs.add(j)

            new_docs.append(words)

            if (j+1)%1000. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)

        empty_docs_index[emotion] = empty_docs
        print ">> %s. %s emotion finishing!" % (i+1,emotion)
        del empty_docs

    pickle.dump(empty_docs_index, open('empty_docs_index.pkl', 'wb'))
    X = BuildFeature.getAvgFeatureVecs(new_docs, model)
    np.savez_compressed(save_path, X=X, y=y)

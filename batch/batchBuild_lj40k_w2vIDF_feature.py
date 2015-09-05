# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from core import BuildFeature, Preprocessing

'''
w2v x idf
'''

def help():
    print "usage: python [idf_type][model_load_path][wordlists_load_path][npz_save_path]"
    print
    print " e.g.: python idf2 /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /corpus/LJ40K/data/pkl/lj40k_wordlists /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+idf2.Xy.npz"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 5: help()
    model = Word2Vec.load(sys.argv[2])
    index2word_set = set(model.index2word)
    num_features = model.syn0.shape[1]

    Words_info = pickle.load(open('/corpus/LJ40K/data/features/tfidf/Words_info_0_800_tf1_tf2_tf3k1b1_idf1_idf2_idf3.pkl','rb'))

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    load_path = sys.argv[3]
    save_path = sys.argv[4]
    idf = sys.argv[1]

    X = np.zeros((40000,num_features),dtype="float32")
    idx = 0
    y = np.array([])

    # emotions = emotions[0:1]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_wordlists.pkl', "rb" ) )

        # docs = docs[0:1]
        for j,doc in enumerate(docs):
            # combine wordlists of one doc in order to get a list of words
            words = []
            for wordlist in doc:
                words += wordlist
            words = Preprocessing.Wordlist_cleaner(words,remove_puncts=True)
            
            featureVec = np.zeros((num_features,),dtype="float32")
            nwords = 0.
            for word in words:
                if word in index2word_set and word in Words_info: 
                    nwords = nwords + Words_info[word][idf]
                    wordvec = Words_info[word][idf]*model[word]
                    featureVec = np.add(featureVec, wordvec)

            featureVec = np.divide(featureVec,nwords)
            # print featureVec.shape
            X[idx] = featureVec
            idx = idx + 1

            if (j+1)%1000. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)
        print ">> %s. %s emotion finishing!" % (i+1,emotion)

    np.savez_compressed(save_path, X=X, y=y)

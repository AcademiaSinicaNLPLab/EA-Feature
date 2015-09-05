# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from core import BuildFeature, Preprocessing

'''
this batch output a pkl file which contain feature information in a dictionary format
e.g.: doc[idx] is {'X': array, 'y': array}
'''
'''
生出lj40k的每個文章的每個句子的w2v representation
'''

def help():
    print "usage: python [model_load_path][pkl_load_path][pkl_save_path(Feature_name folder)]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /corpus/LJ40K/data/pkl/lj40k_wordlists /corpus/LJ40K/data/featurevecs/raw/w2vStanford_rmP_sentences"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 4: help()
    model = Word2Vec.load(sys.argv[1])
    index2word_set = set(model.index2word)

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    load_path = sys.argv[2]

    save_path = sys.argv[3]
    if save_path and not os.path.exists( save_path ): os.makedirs( save_path )
    feature_name = save_path.split("/")[-1]
    
    empty_docs_index = {}
    # emotions = emotions[0:5]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_wordlists.pkl', "rb" ) )
        new_docs_dict = {}
        empty_docs = set()
        for j,doc in enumerate(docs):
            new_doc = []

            for wordlist in doc:
                Use_this_wordlist = False
                wordlist = Preprocessing.Wordlist_cleaner(wordlist,remove_puncts=True)

                if len(wordlist) > 0 :
                    for word in wordlist:
                        if word in index2word_set:
                            Use_this_wordlist = True
                            break
                    if Use_this_wordlist == True:
                        new_doc.append(wordlist)

            if len(new_doc) == 0:
                empty_docs.add(j)
                new_doc = [new_doc]

            X = BuildFeature.getAvgFeatureVecs(new_doc, model)
            y = np.array([emotion]*len(new_doc))
            FeatureVec = {'X': X, 'y': y}
            new_docs_dict[j] = FeatureVec
            if (j+1)%100. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))
            del new_doc
            
        empty_docs_index[emotion] = empty_docs
        pickle.dump(new_docs_dict, open(save_path+'/'+feature_name+'.'+emotion+'.Xy.pkl', 'wb'))
        print ">> %s. %s emotion finishing!" % (i+1,emotion)
        del new_docs_dict
        # print emotion," ",empty_docs
        del empty_docs
    pickle.dump(empty_docs_index, open(save_path+'/empty_docs_index.pkl', 'wb'))


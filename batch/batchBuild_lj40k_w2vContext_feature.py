# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np
from core import BuildFeature, Preprocessing

'''
先把文章切成各個context part
然後把每個context part的每個word的w2v vectors加起來除該context word數，作為這個context part的feature vector
然後concatenate每個part的feature vector成為文章的feature vector

'''
def help():
    print "usage: python [model_load_path][pkl_load_path][number_of_parts][npz_save_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /corpus/LJ40K/data/pkl/lj40k_wordlists 4 /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+OriginalContextInfo4_Maxis_series.Xy.npz"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 5: help()
    model = Word2Vec.load(sys.argv[1])
    index2word_set = set(model.index2word)
    number_of_parts = int(sys.argv[3])

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    load_path = sys.argv[2]
    save_path = sys.argv[4]

    X = np.zeros((40000,300*number_of_parts),dtype="float32")
    y = np.array([])

    idx = 0
    # emotions = emotions[0:1]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_wordlists.pkl', "rb" ) )

        for j,doc in enumerate(docs):
            clean_doc = []
            for wordlist in doc:
                Use_this_wordlist = False
                wordlist = Preprocessing.Wordlist_cleaner(wordlist,remove_puncts=True)
                if len(wordlist) > 0 :
                    for word in wordlist:
                        if word in index2word_set:
                            Use_this_wordlist = True
                            break
                    if Use_this_wordlist == True:
                        clean_doc.append(wordlist)

            n_sentences = len(clean_doc)
            new_doc = []
            if n_sentences < number_of_parts:
                new_doc.append(clean_doc[0])
                for t in xrange(number_of_parts-n_sentences):
                    new_doc.append(clean_doc[0])
                new_doc = new_doc + clean_doc[1:]
            else:
                avg_number_of_sentences = n_sentences / number_of_parts
                number_of_sentences_remain = n_sentences % number_of_parts
                start_row = 0
                for s in xrange(number_of_parts):
                    if number_of_sentences_remain <= 0:
                        Part_of_clean_doc = clean_doc[start_row:start_row+avg_number_of_sentences]
                    else:
                        Part_of_clean_doc = clean_doc[start_row:start_row+avg_number_of_sentences+1]
                    start_row = start_row+len(Part_of_clean_doc)
                    Part_of_clean_doc = sum(Part_of_clean_doc,[])
                    new_doc.append(Part_of_clean_doc)
                    number_of_sentences_remain = number_of_sentences_remain - 1

            DocVec = BuildFeature.getAvgFeatureVecs(new_doc, model)
            DocVec = np.reshape(DocVec, np.product(DocVec.shape))
            X[idx] = DocVec
            idx = idx + 1 

            if (j+1)%100. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))
            del new_doc, clean_doc, DocVec

        label = np.array([emotion]*len(docs))
        y = np.concatenate((y, label), axis=1)        
        del docs
        print ">> %s. %s emotion finishing!" % (i+1,emotion)
    np.savez_compressed(save_path, X=X, y=y)
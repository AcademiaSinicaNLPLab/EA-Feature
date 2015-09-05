# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import csv as csv
import cPickle as pickle
import re
from core import Preprocessing
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def help():
    print "usage: python [raw_data_root][pkl_save_path]"
    print 
    print "WARNING: please modify the code"
    print "sorting_files_on_string for LJ40K; sorting_csvfiles for LJ2M"
    print
    print " e.g.: python /home/bs980201/projects/github_repo/LJ2M/raw/ /corpus/LJ2M/data/pkl/lj2m_sentences"
    print " e.g.: python /home/bs980201/projects/github_repo/LJ40K/raw/ /corpus/LJ40K/data/pkl/lj40k_sentences"
    exit(-1)


if __name__ == '__main__':
    
    if len(sys.argv) != 3: help()

    raw_data_root = sys.argv[1]
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    save_path = sys.argv[2]
    if save_path and not os.path.exists( save_path ): os.makedirs( save_path )

    # emotions = emotions[7:8]
    print emotions
    for i,emotion in enumerate(emotions):
        docs = []
        print ">> %s. %s emotion doc processing..." % (i+1,emotion)
        files = Preprocessing.sorting_files_on_string(raw_data_root, emotion) #for LJ40K
        # files = Preprocessing.sorting_csvfiles(raw_data_root, emotion)# for LJ2M
        for j,f in enumerate(files):
            path = '/'.join([raw_data_root,emotion,f])
            with open(path, 'r') as fr:
                sentences = []
                doc = fr.read()
                doc = doc.decode('utf-8').strip().strip('"')
                doc = Preprocessing.Doc_cleaner(doc, remove_html=True)
                sentences = Preprocessing.Doc_to_Sentences(doc,tokenizer)
                if (j+1)%1000. == 0.:
                    print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(files))
                docs.append(sentences)
        print ">> %s. %s emotion doc finishing!" % (i+1,emotion)        
        pickle.dump(docs, open(save_path+'/'+emotion+'_sentences.pkl', 'wb'))

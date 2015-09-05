# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import cPickle as pickle
from core.stanford import StanfordParser
parser = StanfordParser('../lj2m/englishPCFG.ser.gz')

def help():
    print "usage: jython [raw_data_root][pkl_load_path][pkl_save_path]"
    print
    print " e.g.: jython /home/bs980201/projects/github_repo/LJ2M/raw/ /corpus/LJ2M/data/pkl/lj2m_sentences /corpus/LJ2M/data/pkl/lj2m_wordlists"
    print " e.g.: jython /home/bs980201/projects/github_repo/LJ40K/raw/ /corpus/LJ40K/data/pkl/lj40k_sentences /corpus/LJ40K/data/pkl/lj40k_wordlists"
    exit(-1)

if __name__ == '__main__':

    if len(sys.argv) != 4: help()

    raw_data_root = sys.argv[1]
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])
    load_path = sys.argv[2]
    save_path = sys.argv[3]
    if save_path and not os.path.exists( save_path ): os.makedirs( save_path )

    w2v_sentences = []
    # emotion_sentences = []
    # emotions = emotions[29:]
    for i,emotion in enumerate(emotions):
        new_docs = []
        # docs = [['i am not happy.','.',"i don't like....your car.",'you made.. me happy.'],['i am not happy.','you look good!',"i don't like....your car.",'you made.. me happy.']]
        print 'Start to load '+emotion+'_sentences.pkl'
        docs = pickle.load( open(load_path+'/'+emotion+'_sentences.pkl', "rb" ) )
        for j,sentences in enumerate(docs):
            wordlists = [] 
            for sentence in sentences:
                # split sentence to words
                wordlist = parser.get_tokens(sentence)
                wordlist = [w.lower().encode('utf-8') for w in wordlist]
                if len(wordlist)>=2:
                    wordlists.append(wordlist)
            # emotion_sentences += wordlists
            new_docs.append(wordlists)
            if (j+1)%1000. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,len(docs))
        print 'Start to dump '+emotion+'_wordlists.pkl'
        pickle.dump(new_docs, open(save_path+'/'+emotion+'_wordlists.pkl', 'wb'))
        print ">> %s. %s emotion doc finishing!" % (i+1,emotion)
        print
        del new_docs
        del docs
        # w2v_sentences += emotion_sentences
        # emotion_sentences = []

    # print 'Start to dump all_wordlists.pkl'
    # pickle.dump(w2v_sentences, open('../exp/data/pkl/'+csv_save_name+'/all_wordlists_for_w2v.pkl', 'wb'))
    print 'Finish!!!!!!!!!!!!!!!!!!!'
    



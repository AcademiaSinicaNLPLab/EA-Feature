# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import pymongo
import numpy as np
from pymongo import MongoClient
from gensim.models import Word2Vec
'''
把各個pattern的w2v feature vectors加起來後除pattern數，作為這個文章的feature vector

'''

def help():
    print "usage: python [model_load_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+SumPattern_Maxis_series.Xy.npz"
    exit(-1)


def makeFeatureVec(words, model,neg_pattern=False):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    num_features = model.syn0.shape[1]
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")

    # build negVec
    negVec = np.zeros((num_features,),dtype="float32")
    negVec = np.add(negVec,model["n't"])
    negVec = np.add(negVec,model["not"])
    negVec = np.add(negVec,model["never"])
    negVec = np.divide(negVec,3)

    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    if neg_pattern == True:
        nwords = nwords + 1.
        featureVec = np.add(featureVec,negVec)
    # 
    # Divide the result by the number of words to get the average
    if nwords != 0:
        featureVec = np.divide(featureVec,nwords)
    else:
        print "The input sentence/doc DIDN'T CONTAIN WORDS which are in w2v model."
    
    return featureVec

def getAvgFeatureVecs(docs, model,neg_pattern=False):
    '''Given a set of docs (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    '''
    # Initialize a counter
    counter = 0.
    # 
    num_features = model.syn0.shape[1]
    # Preallocate a 2D numpy array, for speed
    DocFeatureVecs = np.zeros((len(docs),num_features),dtype="float32")
    # 
    # Loop through the docs
    for doc in docs:
       # Print a status message every 1000th doc
       # if counter%1000. == 0.:
       #     print "Review %d of %d" % (counter, len(docs))
       # 
       # Call the function (defined above) that makes average feature vectors
       DocFeatureVecs[counter] = makeFeatureVec(doc, model, neg_pattern)
       #
       # Increment the counter
       counter = counter + 1.

    return DocFeatureVecs



if __name__ == '__main__':
    
    if len(sys.argv) != 3: help()

    model = Word2Vec.load(sys.argv[1])
    num_features = model.syn0.shape[1]
    index2word_set = set(model.index2word)
    y = np.array([])

    ## select mongo collections
    client = MongoClient('doraemon.iis.sinica.edu.tw:27017')
    LJ40K = client.LJ40K
    pats = LJ40K['pats']

    raw_data_root = '/home/bs980201/projects/github_repo/LJ40K/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])

    save_path = sys.argv[2]
    f = open('no_pats_SumPattern.txt', 'w')

    X = np.zeros((40000,num_features),dtype="float32")
    y = np.array([])

    idx = 0
    start = 0
    n_docs_each_emotion = 1000

    # emotions = emotions[3:4]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)

        for j in range(start, start+n_docs_each_emotion):
            DocVec = np.zeros((1,num_features),dtype="float32")
            doc_pats = list(pats.find({"udocID":j}).sort("usentID"))
            n_selected_pats = 0
            for pattern in doc_pats:
                new_wordlist = []
                wordlist = pattern['pattern'].split()

                # let word lower and remove "__" including the word
                neg_pats = False
                for word in wordlist:
                    word = word.encode("utf-8").lower()
                    if '__' in word:
                        word = word.replace('__', '')
                        neg_pats = True
                    new_wordlist.append(word)

                # check whether we can build this pattern vector, if we can't, skip the pattern
                Use_this_wordlist = False
                for word in new_wordlist:
                    if word in index2word_set:
                        Use_this_wordlist = True
                        n_selected_pats = n_selected_pats + 1
                        break

                if Use_this_wordlist == True:
                    PatternVec = getAvgFeatureVecs([new_wordlist], model, neg_pats)
                    DocVec = np.add(DocVec,PatternVec)
                    # print new_wordlist
            if n_selected_pats == 0:
                n_selected_pats = 1
                f.write(emotion+' '+str(j)+'\n')
            DocVec = np.divide(DocVec,n_selected_pats)
            
            X[idx] = DocVec[0]
            idx = idx + 1

            if (j+1)%100. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,n_docs_each_emotion)

        start = start + n_docs_each_emotion
        label = np.array([emotion]*n_docs_each_emotion)
        y = np.concatenate((y, label), axis=1)
        print ">> %s. %s emotion finishing!" % (i+1,emotion)
    f.close()

    # np.savez_compressed(save_path, X=X, y=y)

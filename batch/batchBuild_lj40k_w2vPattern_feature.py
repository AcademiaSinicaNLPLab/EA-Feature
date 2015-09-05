# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import pymongo
import numpy as np
from pymongo import MongoClient
from gensim.models import Word2Vec

'''
把各個pattern的裡面出現的word的vec加起來後除word數，作為這個文章的feature vector

'''
def help():
    print "usage: python [model_load_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context /home/bs980201/projects/github_repo/LJ40K/exp/data/from_mongo/w2vStanford_rmP+Pattern_Maxis_series.Xy.npz"
    exit(-1)


def makeFeatureVec(words, model,neg_pattern=0):
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

    if neg_pattern > 0:
        for i in xrange(neg_pattern):
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
    f = open('no_pats_Pattern.txt', 'w')

    X = np.zeros((40000,num_features),dtype="float32")
    y = np.array([])

    idx = 0
    start = 0
    n_docs_each_emotion = 1000
    # emotions = emotions[0:1]
    for i,emotion in enumerate(emotions):
        print ">> %s. %s emotion processing..." % (i+1,emotion)
        for j in range(start, start+n_docs_each_emotion):
            doc_pats = list(pats.find({"udocID":j}).sort("usentID"))
            new_doc_wordlist = []
            neg_pats = 0
            for pattern in doc_pats:
                wordlist = pattern['pattern'].split()

                # let word lower and remove "__" including the word
                for word in wordlist:
                    word = word.encode("utf-8").lower()
                    if '__' in word:
                        word = word.replace('__', '')
                        neg_pats = neg_pats+1
                    new_doc_wordlist.append(word)

            DocVec = getAvgFeatureVecs([new_doc_wordlist], model, neg_pats)
            X[idx] = DocVec[0]
            idx = idx + 1

            doc_useful = False
            for word in new_doc_wordlist:
                if word in index2word_set:
                    doc_useful = True
                    break

            if doc_useful == False:
                f.write(emotion+' '+str(j%1000)+'\n')
                print emotion,' ', j%1000, ' no pattern useful'


            if (j+1)%100. == 0.:
                print ">>  %s - %d/%d doc finished! " % (emotion,j+1,n_docs_each_emotion)

        start = start + n_docs_each_emotion
        label = np.array([emotion]*n_docs_each_emotion)
        y = np.concatenate((y, label), axis=1)
        print ">> %s. %s emotion finishing!" % (i+1,emotion)


    f.close()
    np.savez_compressed(save_path, X=X, y=y)

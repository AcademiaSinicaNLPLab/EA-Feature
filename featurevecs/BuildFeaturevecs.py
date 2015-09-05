# -*- coding: utf8 -*-
import sys, os
import numpy as np

def makeFeatureVec(words, model):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    num_features = model.syn0.shape[1]
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
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
    # 
    # Divide the result by the number of words to get the average
    if nwords != 0:
        featureVec = np.divide(featureVec,nwords)
    else:
        print "The input sentence/doc DIDN'T CONTAIN WORDS which are in w2v model."
    
    return featureVec

def getAvgFeatureVecs(docs, model):
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
       DocFeatureVecs[counter] = makeFeatureVec(doc, model)
       #
       # Increment the counter
       counter = counter + 1.

    return DocFeatureVecs



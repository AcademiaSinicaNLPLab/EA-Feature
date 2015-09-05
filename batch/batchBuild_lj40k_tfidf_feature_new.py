# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import cPickle as pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import pymongo
from pymongo import MongoClient
from common import utils, filename

def help():
    print "usage: python [tf type][idf type]"
    print
    print " e.g.: python "
    exit(-1)


def build_feature_words_set(globalinfo_path, word_count_limit):
    print 'load words'
    GlobalInfo = pickle.load(open(globalinfo_path,'rb'))
    print 'make word set'
    words_set = set()
    for word in GlobalInfo:
        if GlobalInfo[word]['word_total_count'] >= word_count_limit:
            # print type(word)
            words_set.add(word)

    print len(words_set)
    return words_set


def extract_tfidf_information(collection, words_set, emotions):
    y = np.array([])
    tfidfs = []
    n_sentence_emotion_level = []
    # emotions = emotions[0:5]
    for i,emotion in enumerate(emotions):
        emotion_tfidfs = []
        n_sentence_doc_level = []
        docs = collection.find({"emotion":emotion, "tfidf_type":tfidf_type}).sort("doc_ID")
        # docs = docs[0:3]
        for doc in docs:
            n_sentence_doc_level.append(len(doc['tfidf_value']))
            for sentence_tfidf_values in doc['tfidf_value']:
                new_tfidf_values = {}
                for word, tfidf_value in sentence_tfidf_values:
                    if word.encode('utf-8') in words_set:
                        word = word.encode('utf-8')
                        new_tfidf_values[word] = tfidf_value
                emotion_tfidfs.append(new_tfidf_values)

        n_sentence_emotion_level.append(n_sentence_doc_level)
        tfidfs.append(emotion_tfidfs)
        label = np.array([emotion]*sum(n_sentence_doc_level))
        y = np.concatenate((y, label), axis=1)
        print ">> %s. %s emotion finishing!" % (i+1,emotion)
    return tfidfs, y, n_sentence_emotion_level


def build_tfidf_feature_vecs(v,tfidfs,y,emotions,n_sentence_emotion_level, feature_name, save_path):
    for i,numberlist in enumerate(n_sentence_emotion_level):
        print 'transform data'
        X = v.transform(tfidfs[i])
        start = 0
        X = X.toarray()
        new_docs_dict = {}
        emotion = emotions[i]
        for doc_ID, n in enumerate(numberlist):
            FeatureVec = {'X': X[start:start+n,:], 'y': y[start:start+n]}
            start = start+n
            new_docs_dict[doc_ID] = FeatureVec

        # print new_docs_dict[0]['X'].shape
        # print new_docs_dict[1]['X'].shape
        # print new_docs_dict[2]['X'].shape
        pickle.dump(new_docs_dict, open(save_path+'/'+feature_name+'.'+emotion+'.Xy.pkl', 'wb'))
        print ">> %s. %s emotion finishing!" % (i+1,emotion)

if __name__ == '__main__':
    
    # if len(sys.argv) != 3: help()
    # save_path = sys.argv[3]
    # if save_path and not os.path.exists( save_path ): os.makedirs( save_path )

    tfidf_type = 'tf3idf2_k1p0_b0p8'
    client = MongoClient('doraemon.iis.sinica.edu.tw:27017')
    emotions = filename.emotions['LJ40K']

    db_train = client['LJ2M']
    TFIDF_feature_train = db_train['TFIDF_feature']

    db_test = client['LJ40K']
    TFIDF_feature_test = db_test['TFIDF_feature']
    TFIDF_feature_test_sentence = db_test['TFIDF_feature_sentence']

    words_set = build_feature_words_set('/corpus/LJ2M/data/features/tfidf/GlobalInfo40k.pkl', 300)

    print 'fetch training data from db and extract tfidf info'
    tfidfs_train, y_train, n_sentence_emotion_level_train = extract_tfidf_information(TFIDF_feature_train, words_set, emotions)
    v = DictVectorizer(sparse=True)
    print 'fit training data'
    tfidfs_train_for_fit = sum(tfidfs_train, [])
    v = v.fit(tfidfs_train_for_fit)


    print 'split np.array and dump training data pkl'
    feature_name = tfidf_type
    build_tfidf_feature_vecs(v, tfidfs_train, y_train, emotions, n_sentence_emotion_level_train, feature_name, '/corpus/LJ2M/data/featurevecs/raw/'+feature_name)
    print 'training data ok!'

    print 'fetch testing data from db and extract tfidf info'
    tfidfs_test, y_test, n_sentence_emotion_level_test = extract_tfidf_information(TFIDF_feature_test, words_set, emotions)
    print 'split np.array and dump testing data pkl'
    feature_name = tfidf_type
    build_tfidf_feature_vecs(v, tfidfs_test, y_test, emotions, n_sentence_emotion_level_test, feature_name, '/corpus/LJ40K/data/featurevecs/raw/'+feature_name)
    print 'testing data ok!'


    print 'fetch testing data2 from db and extract tfidf info'
    tfidfs_test, y_test, n_sentence_emotion_level_test = extract_tfidf_information(TFIDF_feature_test_sentence, words_set, emotions)
    print 'split np.array and dump testing data pkl'
    feature_name = tfidf_type+'_sentences'
    build_tfidf_feature_vecs(v, tfidfs_test, y_test, emotions, n_sentence_emotion_level_test, feature_name, '/corpus/LJ40K/data/featurevecs/raw/'+feature_name)
    print 'testing data2 ok!'



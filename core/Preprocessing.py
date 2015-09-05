# -*- coding: utf8 -*-
import sys, os
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

'''
Doc_cleaner -> Doc_to_Sentences ->  Doc_to_Wordlist -> Wordlist_cleaner
'''

def Doc_cleaner(doc, remove_html=True):
    '''remove the noisy part of the document, ex: html text'''
    if remove_html == True:
        try:
            soup = None
            soup = BeautifulSoup(doc)
            
            # brs = soup.find_all('br')
            # for br in brs:
            #     br.replace_with('. ')
            # ps = soup.find_all('p')
            # for p in ps:
            #     p.replace_with('. ')
            
            doc = soup.get_text(" ")
            return doc.strip()
        except:
            return ""
    return doc

def Doc_to_Sentences(doc ,tokenizer):
    '''
    Make a document to a list of sentences
    
    Tips: prepare for the tokenizer when calling this function:
    -------------------------------------------------------------
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    -------------------------------------------------------------
    '''
    sentences = []
    raw_sentences = tokenizer.tokenize(doc)
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(raw_sentence)
    return sentences

def Doc_to_Wordlist(doc, case=3):
    '''
    Make a document or a sentence to a list of words,
    and lower the words.
    '''
    if case == 1:
        #[case1]: only letters
        # e.g.: "32k.. <3" => "'32k','3'"
        doc = re.sub("[^a-zA-Z]"," ", doc)

    elif case == 2:
        #[case2]: letter&number + number + punct
        # e.g.: "32k.. <3" => "'32k', '..', '<','3'"
        words = re.compile(r'(\W+)').split(doc)
        doc = " ".join(words)

    elif case == 3:
        #[case3]: letter&number + number&punct
        # e.g.: "32k.. <3" => "'32k', '..', '<3'"
        words = doc.split()
        p = re.compile(r'(\W+)')
        for i,element in enumerate(words):
            includingwords = re.search(r'[a-zA-Z]',element)
            if includingwords:
                ws = p.split(element)
                words[i] = " ".join(ws)
        doc = " ".join(words)

    words = doc.lower().split()
    return words

def Wordlist_cleaner(words, remove_stopwords=False, remove_puncts=True, remove_numbers=False):
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if remove_puncts:
        punct = [',','.','..']
        words = [w for w in words if not w in punct]

    if remove_numbers:
        words = [w for w in words if not is_number(w)]

    return words

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def sorting_files(root, emotion):
    file_list = [x for x in os.listdir(root + '/' + emotion) if not x.startswith('.')]
    file_list = sorted([int(f) for f in file_list])
    file_list = [str(f) for f in file_list]
    return file_list

def sorting_csvfiles(root, emotion):
    file_list = [x for x in os.listdir(root + '/' + emotion) if not x.startswith('.') and x.endswith('.csv')]
    file_list = sorted([int(f.strip(".csv")) for f in file_list])
    file_list = [ str(f)+".csv" for f in file_list]
    return file_list

def sorting_files_on_string(root, emotion):
    file_list = sorted([x for x in os.listdir(root + '/' + emotion) if not x.startswith('.')])
    return file_list
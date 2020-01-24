import nltk
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re  
import spacy
import string

from spacy.lang.de.examples import sentences 

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


###############
# Replace umlauts for a given text
    
def umlauts(text):
 

    # :param word: text as string
    # :return: manipulated text as str
    
    
    tempVar = text # local variable
    
    # Using str.replace() 
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    
    return tempVar

############

wordnet_lemmatizer = WordNetLemmatizer()

german_stop_words = stopwords.words('german')

german_stop_words_to_use = []   # List to hold words after conversion

# convert umlauts from German stop words list

for word in german_stop_words:
    german_stop_words_to_use.append(umlauts(word))
    
    
    
###################



#define a function for lemmatization with spacy

nlp = spacy.load('de')
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


#################


# Dataset Preprocessing 
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space

def cleaning(df):
    xx=10

    pd.set_option('display.max_columns', None)
    df.rename(columns={'CCB_Liste_AUDI/VW':'label',
                          'MSG_All_Kurzbeschreibung_Anforderung':'Title',
                          'MSG_All_Beschreibung_Anforderung':'CR'}, 
                    inplace=True)
    
    # lemmatization using spacy
    df.CR =  df.apply(lambda x: lemmatizer(x['CR']), axis=1)
    
    #rename labels
    df.label[df.label == 'Allgemeines KonzernCCB'] = 'A'
    df.label[df.label == 'CCB GRA/TSK/Prädiktive Funktionen'] = 'B'
    df.label[df.label == 'CCB Getriebe- und Allradsysteme'] = 'C'
    df.label[df.label == 'CCB Monitoring'] = 'D'
    df.label[df.label == 'Konfigurationen-CCB'] = 'E'
    df.label[df.label == 'KonzernCCB BS/CV/HV'] = 'F'
    df.label[df.label == 'KonzernCCB Brennstoffzelle'] = 'G'
    df.label[df.label == 'KonzernCCB GRA/TSK'] = 'B'
    df.label[df.label == 'KonzernCCB Komponentenfunktion'] = 'I'
    df.label[df.label == 'KonzernCCB Komponentenfunktionen'] = 'I'
    df.label[df.label == 'KonzernCCB PT'] = 'J'
    df.label[df.label == 'KonzernCCB Prädiktive Funktionen'] = 'B'
    
    
   # remove numbers not attached to characters
    df.CR = df.CR.apply(lambda 
                x: re.sub("^\d+\s|\s\d+\s|\s\d+$", "", x))
    df.Title = df.Title.apply(lambda 
                x: re.sub("^\d+\s|\s\d+\s|\s\d+$", "", x))
    
    # replace umlauts for CR descriptions
    df['CR']=df['CR'].apply(umlauts)

    # remove english CRs
    df= df[~(df['CR'].str.contains(' the ') & ~df['CR'].str.contains(' soll '))]
    
    
    # remove punctuations, blank spaces, special characters, and words less than 3 chars
    df['CR'] = df['CR'].apply(lambda x: x.lower())
    df.CR = df.CR.apply(lambda 
                x: x.translate(translator))
    df['CR']= df['CR'].apply(lambda x: re.sub("•“„²", " ", x))
    df['CR']= df['CR'].apply(lambda x: re.sub(r"[\n\r\t]", "", x))
    #df['CR'] = df.CR.apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))
    df['CR'] = df['CR'].apply(lambda x: re.sub(" +", " ", x))
    
    # drop rows with less than xx words
    df['CR_length']=df['CR'].str.split().str.len()
    df = df[df.CR_length >= xx] 
    
    # data tokenazation : split texts into words
    df['CR_tok'] = df.apply(lambda row: nltk.word_tokenize(row['CR']), axis=1)
    
    # remove stop words
    df['CR_tok_s'] = df.apply(lambda row: [item for item in row['CR_tok'] 
                                            if item not in german_stop_words_to_use], axis=1)
    
    df['CR_s'] = df.apply(lambda row: ' '.join(row['CR_tok_s']), axis=1)
    
    
    # data stemming : remove suffixes for german words 
    #df.CR_tok_s = [[sbGer.stem(x) for x in s] for s in df.CR_tok_s]
    
    return df
    
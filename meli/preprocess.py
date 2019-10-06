import keras
import pickle
import pandas as pd
import math
import random
import string
import unicodedata
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

def get_class_distribuicao(df_sample,category_index):
    
    b = random.choice([True,False])
    
    category_name_to_index = {}
    for i in category_index:
        category_name_to_index[category_index[i]]=i
    
    
    ds = df_sample.groupby(['category']).size().reset_index(name='counts')
    dists = []
    for index,row in ds.iterrows():
        dists.append(ds.counts.max()/row['counts'])
    ds['weights'] = dists
    
    class_weight = {}
    for index,row in ds.iterrows():
        if b: class_weight[category_name_to_index[row['category']]] = math.sqrt(row['weights'])
        else: class_weight[category_name_to_index[row['category']]] = 1.0
        
    return class_weight

def get_train_data(train_file,max_length,embedding_file):
    
    #### RUN PREPROCESSING HERE
    
    tokenizer = pickle.load(open("/home/jupyter/fasttext/dicionario.tokenizer",'rb'))
    df_sample = pickle.load(open('/home/jupyter/fasttext/df_sample.data', 'rb'))
    Y = pd.get_dummies(df_sample['category']).values
    number_of_classes = Y.shape[1]
    category_index = pickle.load(open('/home/jupyter/fasttext/category.index', 'rb'))
    X = pickle.load(open('/home/jupyter/fasttext/X.data', 'rb'))
    embedding_matrix = pickle.load(open('/home/jupyter/fasttext/embedding.matrix', 'rb'))
    
    class_weights = get_class_distribuicao(df_sample,category_index)
    
    return X,Y,tokenizer,number_of_classes,category_index,df_sample,max_length,embedding_matrix,class_weights
    
    
def get_test_data(test_file,max_length):
    
    #### RUN PREPROCESSING HERE
    
    tokenizer = pickle.load(open("/home/jupyter/fasttext/dicionario.tokenizer",'rb'))
    df_test = pickle.load(open('/home/jupyter/fasttext/df_test.data', 'rb'))
    category_index = pickle.load(open('/home/jupyter/fasttext/category.index', 'rb'))
    Z = pickle.load(open('/home/jupyter/fasttext/X_test.data', 'rb'))
    
    return Z,tokenizer,category_index,df_test
    


remove_term_codes = False
use_stemming = False


table = str.maketrans({key: None for key in string.punctuation})
stop_words_pt = nltk.corpus.stopwords.words('portuguese')
stop_words_es = nltk.corpus.stopwords.words('spanish')
stop_words_en = nltk.corpus.stopwords.words('english')

stemmer_pt=SnowballStemmer("portuguese")
stemmer_es=SnowballStemmer("spanish")
stemmer_en=SnowballStemmer("english")

STOPWORDS = set(nltk.corpus.stopwords.words('spanish')).union(set(nltk.corpus.stopwords.words('portuguese'))).union(set(nltk.corpus.stopwords.words('english')))


def remove_codes(s):

    if remove_term_codes == True:
        if any(b.isdigit() for b in s)==False: return True
        else: return False
    else:
        return True


def normalize_unit(text):
    tokens = word_tokenize(text)
    s = ''
    for i in range(0,len(tokens)):
        if i == len(tokens): break
        
        w1 = tokens[i]
        
        if i == len(tokens)-1:
            s += w1
            break
            
        w2 = tokens[i+1]
        
        if w1.isdigit() and len(w2) <= 3 and  not(any(b.isdigit() for b in w2)) and not w2 in stop_words_pt and not 2 in stop_words_es and not w2 in stop_words_en:
            s += w1+w2+' '
            tokens[i+1]=''
        else:
            s += w1+' '


    return s

def normalize_title(title):
    title = re.sub('\W+',' ', title)
    return unicodedata.normalize('NFKD', title.lower()).encode('ASCII', 'ignore').decode('utf8')


def clean_text(text,lang):
    text = text.replace('.','')
    text = text.replace(',','')
    text = text.replace('-',' ')
    text = text.replace('/',' ')
    s = normalize_title(text)
    s = normalize_unit(s)
    s = s.translate(table) # remove pontuacao
    tokens = word_tokenize(s) #obtem tokens
    
    v = [i for i in tokens if not i in stop_words_pt and not i in stop_words_es and not i in stop_words_en and not i.isdigit() and len(i) > 1] # remove stopwords

    if use_stemming:
      if lang == 'portuguese':
          v = [stemmer_pt.stem(i) for i in v]
      if lang == 'spanish':
          v = [stemmer_es.stem(i) for i in v]
      if lang == 'english':
          v = [stemmer_en.stem(i) for i in v]    
          
    s = ""
    for t in v: s += t+" "
    text = s.strip()
    

    return text


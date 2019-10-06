from os import listdir
from os.path import isfile, join
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm

voting = {}
def compute_voting(id,label,confidence,model):
    if id not in voting: voting[id] = []
    voting[id].append((label,confidence,model))


def get_prediction_data():

    mypath = './models'
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    L = []
    # getting all predictions by model
    model_predictions = {}
    for f in file_list:
        if '.test' in f:
            print('Reading '+f)
            df = pd.read_csv('./models/'+f)
            df['model'] = [f]*len(df)
            L.append(df)
    df = pd.concat(L)

    
    df.apply(lambda x: compute_voting(x['id'], x['label'], x['confidence'], x['model']), axis=1)
    
    for id in voting:
        voting[id].sort(key=lambda tup: tup[1],reverse=True)
    
    return voting


def generate_network(df_train, df_test, voting):
    
    label_to_code = {}
    code_to_label = {}
    label_counter = 0
    for category in df_train.category.unique():
        label_to_code[category]=label_counter
        code_to_label[label_counter]=category
        label_counter+=1
    
    G = nx.Graph()

    # network with train data
    for index,row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        tokens = str(row['title_clean']).strip().split(' ')
        doc = str(index)+':doc'

        for token in tokens:
            G.add_edge(doc,token+':term')

        if len(tokens) > 1:
            G.nodes[doc]['y'] = np.array([0.0]*len(label_to_code))
            G.nodes[doc]['y'][label_to_code[row['category']]]=1.0


    # network with test data
    for index,row in tqdm(df_test.iterrows(),total=df_test.shape[0]):
        tokens = str(row['title_clean']).strip().split(' ')
        doc = str(index)+':doc_test'

        if len(tokens) > 1:

            for token in tokens:
                G.add_edge(doc,token+':term')    


    # network with models

    min_classifiers = 5
    confidence = 0.9

    counter = 0
    for id in tqdm(voting):
        L = voting[id]
        counter = 0
        node_doc = str(id)+':doc_test'
        for vote in L:
            if vote[1] >= confidence:
                label = vote[0]
                node_model = vote[0]+'_'+vote[2]+':model'
                G.add_edge(node_doc,node_model)
                G.nodes[node_model]['y'] = np.array([0.0]*len(label_to_code))
                G.nodes[node_model]['y'][label_to_code[label]]=1.0
                counter += 1

        if counter < min_classifiers:
            counter = 0
            for vote in L:
                label = vote[0]
                node_model = vote[0]+'_'+vote[2]+':model'
                G.add_edge(node_doc,node_model)
                G.nodes[node_model]['y'] = np.array([0.0]*len(label_to_code))
                G.nodes[node_model]['y'][label_to_code[label]]=1.0
                counter+=1
                if counter > min_classifiers: break

    return G, label_to_code, code_to_label



cacheU = {}
def get_degree(G,node):
    if node in cacheU: return cacheU[node]
    counter = 0.0
    for n in G.neighbors(node):
        counter += 1.0
    
    cacheU[node]=counter
    
    return counter


def regularization(G, code_to_label):
    
    for node in tqdm(G.nodes()):
        if 'y' in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node]['y']*1
        else:
            G.nodes[node]['f'] = np.array([0.0]*len(code_to_label))
            


    for i in tqdm(range(1,15)):
        #print('Iteration '+str(i))
        for node in G.nodes():
            if 'y' not in G.nodes[node]:
                f_new = np.array([0.0]*len(code_to_label))
                w_sum = 0.0
                for neighbor in G.neighbors(node):
                    degree = get_degree(G,neighbor)
                    w = 1.0/np.sqrt(degree)
                    w_sum += w
                    f_new += w*G.nodes[neighbor]['f']
                f_new /= w_sum
                G.nodes[node]['f']=f_new


                
    return G



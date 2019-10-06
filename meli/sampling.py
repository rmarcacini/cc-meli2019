from tqdm import tqdm
import pandas as pd


def balanced_sampling(df, num_instances=500):


    L = []
    counter = 1
    categories = df['category'].unique()
    for category in tqdm(categories):
        sample = df[df.category==category]
        try:
            sample_reliable = sample[sample.label_quality=='reliable'].sample(num_instances,replace=True)
            L.append(sample_reliable)
        except:
            x=1

        sample_unreliable = sample[sample.label_quality=='unreliable'].sample(num_instances,replace=True)
        L.append(sample_unreliable)
        counter+=1


    return pd.concat(L)


def unbalanced_sampling(df, num_instances=1000000):


    num_classes = len(df['category'].unique())
    
    while True:
        sample = df.sample(num_instances)
        if len(sample['category'].unique()) == num_classes:
            return sample
    
    
import pandas as pd
from collections import Counter
import numpy as np
import pickle
from string import punctuation

df=pd.read_csv("./Intermediate Files/Data.csv")

df.drop('Unnamed: 0', axis=1, inplace=True)


df['text'] = df['text'].apply(lambda x: x.rstrip())
df['text']=df['text'].astype(str)

s_text=" ".join([i for i in df['text'] if i not in punctuation])

sentences = s_text.split(' ')

counts = Counter(sentences)

vocab = sorted(counts, key = counts.get, reverse = True)

vocab_to_int = {word: li for li, word in enumerate(vocab,1)}

with open('./Intermediate Files/vocab_dictionary.pickle', 'wb') as handle:
    pickle.dump(vocab_to_int, handle)
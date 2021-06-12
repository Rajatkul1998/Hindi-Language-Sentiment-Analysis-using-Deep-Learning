import pandas as pd
import codecs



df=pd.read_csv("./Intermediate Files/Data.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)


df['text'] = df['text'].apply(lambda x: x.rstrip())
df['text']=df['text'].astype(str)


def convert_to_list(strn):
    temp=[]
    l=strn.split(" ")
    for word in l:
            temp.append(word)
    return temp 
df['text']=df['text'].apply(lambda x:convert_to_list(x))
train_df = df.iloc[:2000,:] 
test_df= df.iloc[2001:,:]

train_df.to_csv("./Intermediate Files/Train.csv")
test_df.to_csv("./Intermediate Files/Test.csv")
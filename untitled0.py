import numpy as np
import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import pickle
import string

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import streamlit as st
from io import StringIO


df_train = pd.read_csv(r'C:/Users/r.zhanabai/Documents/project/training_text.zip', engine='python', sep='\|\|', skiprows=1, names=["ID", "Text"]).set_index('ID')
df_train2 = pd.read_csv(r'C:/Users/r.zhanabai/Documents/project/training_variants.zip').set_index('ID')
train = pd.merge(df_train2, df_train, how='inner', on='ID').fillna('')

    
dataset = st.file_uploader("upload first", type = ['csv'])
if dataset is not None:
    df_test = pd.read_csv(dataset, engine='python', sep='\|\|', header=None, skiprows=1, names=["ID", "Text"]).set_index('ID')
    st.write(df_test)
    
dataset2 = st.file_uploader("upload second", type = ['csv'])
if dataset is not None:
    df_test2 = pd.read_csv(dataset2).set_index('ID')
    st.write(df_test2)  
    
test = pd.merge(df_test, df_test2, how='inner', on='ID').fillna('')

(train['Class'].value_counts(sort=False) / train.shape[0]).plot(kind='bar')
plt.plot()

stop_words = set(stopwords.words('english'))

def preprocessing(text):
    global stop_words
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
   
    return text

tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=500)

text_train = tfidf.fit_transform(train['Text'].values).toarray()
text_test = tfidf.transform(test['Text'].values).toarray()

train2 = pd.DataFrame(text_train, index=train.index)  
test2 = pd.DataFrame(text_test, index=test.index)

n_components = 70
svd_truncated = TruncatedSVD(n_components=n_components, n_iter=40, random_state=42)
truncated_train = pd.DataFrame(svd_truncated.fit_transform(train2))
truncated_test = pd.DataFrame(svd_truncated.transform(test2))

truncated_train.columns = truncated_test.columns = [f'component №{i}' for i in range(1, n_components + 1)]

all_data = pd.concat([train, test]).reset_index(drop=True)
all_data = pd.get_dummies(all_data, columns=['Gene', 'Variation'], drop_first=True)
all_data.drop('Text', axis=1, inplace=True)



train = all_data.loc[train.index]

ind = sorted(set(all_data.index) - set(train.index))
test = all_data.loc[ind]

truncated_test.index = ind

train = train.join(truncated_train)
test = test.join(truncated_test)

X = train.drop('Class', axis=1)
Y = train['Class'].values - 1

X_test = test.drop('Class', axis=1)


X_train, Y_train = X.copy(), Y.copy()

matrix_test = xgb.DMatrix(X_test)

pickled_model = pickle.load(open(r'C:\Users\r.zhanabai\Documents\project\model.pkl', 'rb'))

pred = pickled_model.predict(matrix_test)

submit = pd.DataFrame(pred, columns=[f'class{i}' for i in range(1, 10)])
submit.insert(loc=0, column='ID', value=pd.merge(df_test2, df_test, how='inner', on='ID').fillna('').index)
def convert_df(submit):
   return submit.to_csv(index=False).encode('utf-8')


csv = convert_df(submit)

st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

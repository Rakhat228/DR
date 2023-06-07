import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import pickle

import seaborn as sns
sns.set()
import streamlit as st


def classify():
    
    df_train = pd.read_csv(r'C:/Users/r.zhanabai/Documents/project/training_text.zip', engine='python', sep='\|\|', skiprows=1, names=["ID", "Text"]).set_index('ID')
    df_train2 = pd.read_csv(r'C:/Users/r.zhanabai/Documents/project/training_variants.zip').set_index('ID')
    train = pd.merge(df_train2, df_train, how='inner', on='ID').fillna('')
    
    test = pd.merge(df_test, df_test2, how='inner', on='ID').fillna('')
    
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
     
    X_test = test.drop('Class', axis=1)

    
    matrix_test = xgb.DMatrix(X_test)
    
    
    pickled_model = pickle.load(open(r'C:\Users\r.zhanabai\Documents\project\model.pkl', 'rb'))
    
    pred = pickled_model.predict(matrix_test)
    
    submit = pd.DataFrame(pred, columns=[f'class{i}' for i in range(1, 10)])
    submit.insert(loc=0, column='ID', value=pd.merge(df_test2, df_test, how='inner', on='ID').fillna('').index)
    def convert_df(df_test2):
       return df_test2.to_csv(index=False).encode('utf-8')
    
    
    max_class_value = []
    for j in range(len(pred)):
        for i in range(9):
            if max(pred[j]) == pred[j][i]:
                max_class_value.append(i+1)
    df_test2['Class'] = max_class_value
    
    csv = convert_df(df_test2)
    
    st.download_button(
       "Press to Download",
       csv,
       "Output.csv",
       "text/csv",
       key='download-csv'
    )
     
    
    return (st.write(submit) , st.write(df_test2), convert_df(df_test2))


dataset = st.file_uploader("UPLOAD TEXT FILE", type = ['csv'])
if dataset is not None:
    df_test = pd.read_csv(dataset, engine='python', sep='\|\|', header=None, skiprows=1, names=["ID", "Text"]).set_index('ID')
    st.write(df_test)
    
dataset2 = st.file_uploader("UPLOAD VARIANTS FILE", type = ['csv'])
if dataset2 is not None:
    df_test2 = pd.read_csv(dataset2).set_index('ID')
    st.write(df_test2)  

st.button('Classify', on_click=classify, disabled=False)

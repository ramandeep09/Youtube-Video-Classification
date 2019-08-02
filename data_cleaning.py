# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:36:02 2019

@author: Raman
"""

import pandas as pd

df_copy = pd.read_csv('final_data.csv')

df_link = pd.DataFrame(columns = ["link"])        
df_title = pd.DataFrame(columns = ["title"])        
df_description = pd.DataFrame(columns = ["description"])        
df_category = pd.DataFrame(columns = ["category"])        
df_link['link'] = df_copy['link'] 
df_title ['title']= df_copy['title'] 
#df_copy['description'] .dropna(axis=0, how='all',inplace = True)

df_description['description'] = df_copy['description'] 
df_category['category'] = df_copy['category']
df_description.dropna(axis=0, how='all',inplace = True)
df_description.reset_index(drop=True, inplace=True)
import re 
import nltk 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

corpus = []        
for i in range(0, 3278):         
  review = re.sub('[^a-zA-Z]', ' ', df_title['title'][i])
  review = review.lower()            
  review = review.split()            
  ps = PorterStemmer()            
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]            
  review = ' '.join(review)            
  corpus.append(review)
"""  
loc = pd.DataFrame(columns = ["location"])
for i in range(0,3278):
    loc['location'] = str(df_description[i])
"""
corpus1 = [] 
for i in range(0, 3142):            
  review = re.sub('[^a-zA-Z]', ' ', df_description['description'][i])
  review = review.lower()            
  review = review.split()            
  ps = PorterStemmer()            
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]            
  review = ' '.join(review)            
  corpus1.append(review)


dftitle = pd.DataFrame({'title':corpus})
dfdescription = pd.DataFrame({'description':corpus1})  


from sklearn.preprocessing import LabelEncoder
dfcategory = df_category.apply(LabelEncoder().fit_transform)
df_new = pd.concat([df_link, dftitle, dfdescription, dfcategory], axis=1, join_axes = [df_link.index])

from sklearn.feature_extraction.text import CountVectorizer   
cv = CountVectorizer(max_features=1500) 
X = cv.fit_transform(corpus,corpus1).toarray() 
y = df_new.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


"""
from sklearn.linear_model import LinearRegression

regresser = LinearRegression()
regresser.fit(X_train,y_train)
regresser.score(X_test,y_test)

y_pred1 = regresser.predict(X_test)
"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
"""

import seaborn as sns
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actusal Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Value" ,ax=ax1)

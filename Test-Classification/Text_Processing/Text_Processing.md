
Import and play around with the Data:


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import *
```


```python
full = pd.read_csv("/Users/1/Desktop/train.csv")
full = full.rename(columns={"V1":"type", "V2":"sms"})
full['type'] = full['type'].apply(lambda x: 1 if x =='spam' else 0)
full.head()
full.type.value_counts()
```




    0    3862
    1     598
    Name: type, dtype: int64




```python
# I did it in matlab, lets make histogram here too
import seaborn as sb
sb.countplot(full['type'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a116c8d68>




![png](output_3_1.png)



```python
cleaned = full.copy()
cleaned.sms = cleaned.sms.str.replace('!', ' ')
cleaned.sms = cleaned.sms.str.replace('#', ' ')
cleaned.sms = cleaned.sms.str.replace('$', ' ')
cleaned.sms = cleaned.sms.str.replace('%', ' ')
cleaned.sms = cleaned.sms.str.replace('&', ' ')
cleaned.sms = cleaned.sms.str.replace('(', ' ')
cleaned.sms = cleaned.sms.str.replace(')', ' ')
cleaned.sms = cleaned.sms.str.replace('*', ' ')                                
cleaned.sms = cleaned.sms.str.replace('+', ' ')
cleaned.sms = cleaned.sms.str.replace('-', ' ')                                
cleaned.sms = cleaned.sms.str.replace('.', ' ')                                
cleaned.sms = cleaned.sms.str.replace('\\', ' ')  
cleaned.sms = cleaned.sms.str.replace('"', ' ')  
cleaned.sms = cleaned.sms.str.replace('/', ' ')
cleaned.sms = cleaned.sms.str.replace(':', ' ')
cleaned.sms = cleaned.sms.str.replace(';', ' ')
cleaned.sms = cleaned.sms.str.replace('<', ' ')
cleaned.sms = cleaned.sms.str.replace('=', ' ')
cleaned.sms = cleaned.sms.str.replace('>', ' ')                                
cleaned.sms = cleaned.sms.str.replace('?', ' ')                                
cleaned.sms = cleaned.sms.str.replace('@', ' ')
cleaned.sms = cleaned.sms.str.replace('[', ' ')
cleaned.sms = cleaned.sms.str.replace(']', ' ')
cleaned.sms = cleaned.sms.str.replace('^', ' ')
cleaned.sms = cleaned.sms.str.replace('_', ' ')
cleaned.sms = cleaned.sms.str.replace('{', ' ')
cleaned.sms = cleaned.sms.str.replace('|', ' ')
cleaned.sms = cleaned.sms.str.replace('}', ' ')
cleaned.sms = cleaned.sms.str.replace('~', ' ')
cleaned.sms = cleaned.sms.str.replace(',', ' ')
cleaned.sms = cleaned.sms.str.replace('\'', ' ')
cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Ok lar    Joking wif u oni</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>U dun say so early hor    U c already then say</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>FreeMsg Hey there darling it s been 3 week s n...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Even my brother is not like to speak with me  ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#1. Get rid of numbers. There is probably an easier way to do that, but i don't have internet
cleaned.sms = cleaned.sms.str.replace('0', '')
cleaned.sms = cleaned.sms.str.replace('1', '')
cleaned.sms = cleaned.sms.str.replace('2', '')
cleaned.sms = cleaned.sms.str.replace('3', '')
cleaned.sms = cleaned.sms.str.replace('4', '')
cleaned.sms = cleaned.sms.str.replace('5', '')
cleaned.sms = cleaned.sms.str.replace('6', '')
cleaned.sms = cleaned.sms.str.replace('7', '')
cleaned.sms = cleaned.sms.str.replace('8', '')
cleaned.sms = cleaned.sms.str.replace('9', '')
#2. Get rid of capitol letters
cleaned.sms = cleaned.sms.str.lower()
#3 Get rid of extra space
cleaned.sms = cleaned.sms.str.replace("     ", ' ')
cleaned.sms = cleaned.sms.str.replace("     ", ' ')
cleaned.sms = cleaned.sms.str.replace("    ", ' ')
cleaned.sms = cleaned.sms.str.replace("   ", ' ')
cleaned.sms = cleaned.sms.str.replace("  ", ' ')
```


```python
#tokenizing messages
cleaned['word_tokens'] = cleaned.apply(lambda x: x['sms'].split(' '), axis=1)
```


```python

from nltk.corpus import stopwords
#remove stopwords
cleaned['wout_stopwords'] = cleaned.apply(lambda x: [word for word in x['word_tokens'] if word not in stopwords.words('english')], axis=1)
#stemming
ps = PorterStemmer()
cleaned['stemmed'] = cleaned.apply(lambda x: [ps.stem(word) for word in x['wout_stopwords']], axis = 1)
#remove 2-letter words
cleaned['cleaned_final'] = cleaned.apply(lambda x: ' '.join([word for word in x['stemmed'] if len(word) > 2]), axis = 1)
cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>sms</th>
      <th>word_tokens</th>
      <th>wout_stopwords</th>
      <th>stemmed</th>
      <th>cleaned_final</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>ok lar joking wif u oni</td>
      <td>[ok, lar, joking, wif, u, oni, ]</td>
      <td>[ok, lar, joking, wif, u, oni, ]</td>
      <td>[ok, lar, joke, wif, u, oni, ]</td>
      <td>lar joke wif oni</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>free entry in a wkly comp to win fa cup final ...</td>
      <td>[free, entry, in, a, wkly, comp, to, win, fa, ...</td>
      <td>[free, entry, wkly, comp, win, fa, cup, final,...</td>
      <td>[free, entri, wkli, comp, win, fa, cup, final,...</td>
      <td>free entri wkli comp win cup final tkt may tex...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>u dun say so early hor u c already then say</td>
      <td>[u, dun, say, so, early, hor, u, c, already, t...</td>
      <td>[u, dun, say, early, hor, u, c, already, say, ]</td>
      <td>[u, dun, say, earli, hor, u, c, alreadi, say, ]</td>
      <td>dun say earli hor alreadi say</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>freemsg hey there darling it s been week s now...</td>
      <td>[freemsg, hey, there, darling, it, s, been, we...</td>
      <td>[freemsg, hey, darling, week, word, back, like...</td>
      <td>[freemsg, hey, darl, week, word, back, like, f...</td>
      <td>freemsg hey darl week word back like fun still...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>even my brother is not like to speak with me t...</td>
      <td>[even, my, brother, is, not, like, to, speak, ...</td>
      <td>[even, brother, like, speak, treat, like, aids...</td>
      <td>[even, brother, like, speak, treat, like, aid,...</td>
      <td>even brother like speak treat like aid patent</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create term frequency matrix
from sklearn.feature_extraction.text import CountVectorizer
freq = pd.Series(' '.join(cleaned['cleaned_final']).split()).value_counts()[:150]
freq = list(freq.index)
cleaned['new'] = cleaned['cleaned_final'].apply(lambda x: " ".join(ch for ch in x.split() if ch in freq))
vec = CountVectorizer()
X = vec.fit_transform(cleaned.new)
df_words = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df_words.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alreadi</th>
      <th>also</th>
      <th>alway</th>
      <th>amp</th>
      <th>anyth</th>
      <th>around</th>
      <th>ask</th>
      <th>award</th>
      <th>babe</th>
      <th>back</th>
      <th>...</th>
      <th>well</th>
      <th>win</th>
      <th>wish</th>
      <th>word</th>
      <th>work</th>
      <th>would</th>
      <th>www</th>
      <th>yeah</th>
      <th>year</th>
      <th>yet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 150 columns</p>
</div>




```python
#add number of characters as variable
#df_words['nchars'] = full.sms.str.len()
# or the same using a lambda function, man they're dope
df_words['nchars'] = full['sms'].apply(lambda x: len([ch for ch in x]))
#add number of symbols 
df_words['nsymbs'] = full['sms'].apply(lambda x: len( [ch for ch in x if not ch.isalpha() and ch != ' '] ))
#add number of letters in caps
df_words['ncaps'] = full['sms'].apply(lambda x: len([ch for ch in x if ch.isupper()]))
```


```python
#data partitioning, scaling/centering
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df_words,full['type'], test_size = 0.2, random_state = 1)
from sklearn import preprocessing
x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
```


```python
from sklearn import svm
svm_train = svm.SVC()
svm_model = svm_train.fit(x_train, y_train)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, svm_model.predict(x_test)))
```

    [[753   0]
     [ 17 122]]



```python
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(svm_model, x_train, y_train, cv=10)
y_scores = cross_val_predict(svm_model, x_test, y_test, cv=10)
```

    /Users/1/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
print("Cross-validated scores:", scores)
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_scores))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)
from sklearn.metrics import roc_curve, auc
roc_auc = auc(fpr, tpr)
print("Area Under Curve: ",auc(fpr, tpr))


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


```

    Cross-validated scores: [0.96078431 0.95798319 0.97478992 0.9719888  0.9719888  0.98319328
     0.9859944  0.96918768 0.96078431 0.98591549]
    Confusion Matrix:
     [[752   1]
     [ 35 104]]
    Area Under Curve:  0.8734367088002905



![png](output_13_1.png)



```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
tree_model = clf.fit(x_train, y_train)
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, tree_model.predict(x_test)))
```

    Confusion Matrix:
     [[746   7]
     [ 19 120]]



```python
#cross validate
scores = cross_val_score(tree_model, x_train, y_train, cv=10)
y_scores = cross_val_predict(tree_model, x_test, y_test, cv=10)
print("Cross-validated scores:", scores)
#accuracy based on cross validation
print("Confusion Matrix, cross validated:\n", metrics.confusion_matrix(y_test, y_scores))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)
print("Area Under Curve: ",auc(fpr, tpr))
```

    Cross-validated scores: [0.96078431 0.95238095 0.9719888  0.9719888  0.95518207 0.96638655
     0.9719888  0.96638655 0.96638655 0.96901408]
    Confusion Matrix, cross validated:
     [[730  23]
     [ 18 121]]
    Area Under Curve:  0.9199795542052414



```python
from sklearn.neighbors import KNeighborsClassifier
# instantiate learning model (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=3)
# fitting the model
knn_model.fit(x_train, y_train)
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, knn_model.predict(x_test)))
```

    Confusion Matrix:
     [[749   4]
     [ 54  85]]



```python
scores = cross_val_score(knn_model, x_train, y_train, cv=10)
y_scores = cross_val_predict(knn_model, x_test, y_test, cv=10)
print("Cross-validated scores:", scores)
#accuracy based on cross validation
print("Confusion Matrix, cross validated:\n", metrics.confusion_matrix(y_test, y_scores))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)
print("Area Under Curve: ",auc(fpr, tpr))
```

    Cross-validated scores: [0.93837535 0.93557423 0.96358543 0.9719888  0.94117647 0.95518207
     0.98319328 0.94117647 0.93557423 0.95774648]
    Confusion Matrix, cross validated:
     [[747   6]
     [ 54  85]]
    Area Under Curve:  0.8017713319384333



```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
lda_model = clf.fit(x_train, y_train)
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, lda_model.predict(x_test)))
```

    Confusion Matrix:
     [[752   1]
     [ 19 120]]



```python
#cross validate
scores = cross_val_score(lda_model, x_train, y_train, cv=10)
y_scores = cross_val_predict(lda_model, x_test, y_test, cv=10)
print("Cross-validated scores:", scores)
#accuracy based on cross validation
print("Confusion Matrix, cross validated:\n", metrics.confusion_matrix(y_test, y_scores))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)
print("Area Under Curve: ",auc(fpr, tpr))
```

    Cross-validated scores: [0.96358543 0.96358543 0.96918768 0.9859944  0.96638655 0.97759104
     0.99159664 0.96358543 0.96638655 0.98591549]
    Confusion Matrix, cross validated:
     [[747   6]
     [ 20 119]]
    Area Under Curve:  0.9240734902118146


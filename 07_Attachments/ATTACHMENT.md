import os

import pandas as pd

import nltk

import re

from google.colab import drive

drive.mount('drive')

from zipfile import ZipFile

path = '/content/drive/MyDrive/archive.zip'

extracted_files = '/content/aclImdb'

with ZipFile(path, 'r') as z_file:

    z_file.extractall(extracted_files)

def loading(directory):

  data = []

  for label in ['pos']:

    folder = os.path.join(directory, label)

    for file in os.listdir(folder):

      if file.endswith('.txt'):

        with open(os.path.join(folder, file), 'r', encoding = 'utf-8') as f:

          text = f.read()

          data.append([text, 1 if label == 'pos' else 0])

    for label in ['neg']:

      folder = os.path.join(directory, label)

      for file in os.listdir(folder):

        if file.endswith('.txt'):

          with open(os.path.join(folder, file), 'r', encoding = 'utf-8') as f:

            text = f.read()

            data.append([text, 0 if label == 'neg' else 1])

    return pd.DataFrame(data, columns = ['text', 'label'])

training_sample = loading(os.path.join(extracted_files, 'aclImdb/train'))

test_sample = loading(os.path.join(extracted_files, 'aclImdb/test'))

training_sample = pd.concat([training_sample, test_sample], axis=0, join='inner')

print(training_sample)

nltk.download("stopwords")

nltk.download('punkt')

nltk.download('punkt_tab')

nltk.download('wordnet')

lemmatize = nltk.WordNetLemmatizer()

from nltk.corpus import stopwords

def preprocess_text(text):

  text = re.sub("[^a-zA-Z]"," ",text)

  text = nltk.word_tokenize(text, language = "english")

  text = [lemmatize.lemmatize(word) for word in text if not word in set(stopwords.words("english"))]

  text = " ".join(text)

  return text.replace('br br', ' ')

training_sample['new_text'] = training_sample['text'].apply(preprocess_text)

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

Text = " ".join(training_sample['new_text'])

wordcloud = WordCloud().generate(Text)

plt.axis("off")

plt.imshow(wordcloud)

print(training_sample['new_text'])

print(training_sample['label'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

X_train = tfidf.fit_transform(training_sample['new_text'])

Y_train = training_sample['label'].values

print(X_train[:10])

print(Y_train[:10])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.33, random_state = 42)

import numpy as np

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

x_train_a = (x_train).toarray()

x_test_a = (x_test).toarray()

x = np.array(x_train_a)

y = np.array(y_train)

x_t = np.array(x_test_a)

y_t = np.array(y_test)

result_bayes = nb.fit(x, y)

nb.score(x_t, y_t)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

result_logreg = logreg.fit(x_train, y_train)

logreg.score(x_test, y_test)

from sklearn import svm

methodsvm = svm.SVC()

result_svm = methodsvm.fit(x_train[:1001], y_train[:1001])

methodsvm.score(x_test[:1001], y_test[:1001])

Y_pred_nb = nb.predict(x_test_a)

Y_pred_nb = (Y_pred_nb >= 0.5).astype("int")

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(classification_report(y_test, Y_pred_nb))

print(confusion_matrix(y_test, Y_pred_nb))

from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_nb)

plt.plot(fpr, tpr)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

auc = metrics.roc_auc_score(y_test, Y_pred_nb)

print("AUC: %.3f" % auc)

from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sns

cm = confusion_matrix(y_test, Y_pred_nb)

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix - Naive Bayes')

plt.show()

Y_pred_lr = logreg.predict(x_test)

Y_pred_lr = (Y_pred_lr >= 0.5).astype("int")

print(classification_report(y_test, Y_pred_lr))

print(confusion_matrix(y_test, Y_pred_lr))

fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_lr)

plt.plot(fpr, tpr)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

auc = metrics.roc_auc_score(y_test, Y_pred_lr)

print("AUC: %.3f" % auc)

cm = confusion_matrix(y_test, Y_pred_lr)

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix - Logistic Regression')

plt.show()

Y_pred_svm = methodsvm.predict(x_test)

Y_pred_svm = (Y_pred_svm >= 0.5).astype("int")

print(classification_report(y_test, Y_pred_svm))

print(confusion_matrix(y_test, Y_pred_svm))

fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_svm)

plt.plot(fpr, tpr)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

auc = metrics.roc_auc_score(y_test, Y_pred_svm)

print("AUC: %.3f" % auc)

cm = confusion_matrix(y_test, Y_pred_svm)

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix - SVM')

plt.show()
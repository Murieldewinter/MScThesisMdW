# %%
# installing required packages

import numpy as np
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# %%
# reading the training dataset
file_path = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/EXIST_2021_Dataset/training/EXIST2021_training.tsv'

# Read the .tsv file
Sexism_complete_train = pd.read_csv(file_path, delimiter='\t')

Sexism_train= Sexism_complete_train[(Sexism_complete_train['language']=='es')& (Sexism_complete_train['source']=='twitter')]
texts_es = Sexism_train['text'].tolist()
for text in texts_es[:20]:
  print(text)

file_path_es = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/training/texts_es.tsv'


# %%

# Read the testing dataset
file_path = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/test/EXIST2021_test_labeled.tsv'
Sexism_complete_test = pd.read_csv(file_path, delimiter='\t')


Sexism_test= Sexism_complete_test[(Sexism_complete_test['language']=='es')& (Sexism_complete_test['source']=='twitter')]
texts_es = Sexism_test['text'].tolist()
for text in texts_es[:20]:
  print(text)

file_path_es = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/test/texts_es.tsv'



# %%
# retrieving the SBW embeddings for words train dataset
file_path = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/SBW-vectors-300-min5.bin.gz'
W2Vembeddings = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
text_column = Sexism_train['text']
all_embeddings = []
for text in text_column:
  words = text.split()
  row_embeddings = []
  for word in words:
    if word in W2Vembeddings:
      embedding = W2Vembeddings[word]
      row_embeddings.append(embedding)
  if row_embeddings:
    row_embedding_avg = np.mean(row_embeddings, axis=0)
  else:
    # Use a zero vector of the same length as other embeddings
    row_embedding_avg = np.zeros(W2Vembeddings.vector_size)

  all_embeddings.append(row_embedding_avg)
# adding the SBW embeddings as a column to the train dataset
Sexism_train['embeddings'] = all_embeddings

# averaging the word embeddings to get the sentence embeddings
Sexism_train['sentence_embedding'] = Sexism_train['embeddings'].apply(lambda x: np.mean(x, axis=0))
print(Sexism_train['sentence_embedding'].head())

# %%
print(Sexism_train['sentence_embedding'])

# %%
# Retrieving SBW embeddings for test dataset
text_column = Sexism_test['text']
all_embeddings = []
for text in text_column:
  words = text.split()
  row_embeddings = []
  for word in words:
    if word in W2Vembeddings:
      embedding = W2Vembeddings[word]
      row_embeddings.append(embedding)
  if row_embeddings:
    row_embedding_avg = np.mean(row_embeddings, axis=0)
  else:
    # Use a zero vector of the same length as other embeddings
    row_embedding_avg = np.zeros(W2Vembeddings.vector_size)

  all_embeddings.append(row_embedding_avg)
# adding the SBW embeddings as a column to the test dataset
Sexism_test['embeddings'] = all_embeddings

# averaging the word embeddings to get the sentence embeddings
Sexism_test['sentence_embedding'] = Sexism_test['embeddings'].apply(lambda x: np.mean(x, axis=0))

# %%
print(Sexism_test['sentence_embedding'])

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# SVM Classifer
X_test = np.array(Sexism_test['embeddings'].tolist())
y_test = Sexism_test['task1'].values

X_train = np.array(Sexism_train['embeddings'].tolist())  # Convert list of lists to numpy array
y_train = Sexism_train['task1'].values

# Create an SVM with a pipeline that includes scaling
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Fit the model
clf.fit(X_train, y_train)

# Predict and evaluate
predictions = clf.predict(X_test)

# %%
# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# Scores
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='binary', pos_label='sexist')  # Adjust according to your labels
recall = recall_score(y_test, predictions, average='binary', pos_label='sexist')        # Adjust according to your labels
f1 = f1_score(y_test, predictions, average='binary', pos_label='sexist')               # Adjust according to your labels

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")



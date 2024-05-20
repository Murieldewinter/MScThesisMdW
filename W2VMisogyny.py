# %%

import numpy as np
import pandas as pd
import msoffcrypto
import io
import gensim
import openpyxl
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV


file_path = '/home/u161198/new_env/Datasets/es_AMI_TrainingSet_NEW (2).xlsx'
password = '!20AMI_ES18?'

file = msoffcrypto.OfficeFile(open(file_path, 'rb'))
file.load_key(password=password)

decrypted_stream = io.BytesIO()
file.decrypt(decrypted_stream)

decrypted_stream.seek(0)
Misogyny_train = pd.read_excel(decrypted_stream, engine='openpyxl')

print(Misogyny_train.head)



# %%
file_path = '/home/u161198/new_env/Datasets/es_AMI_TestSet.xlsx'
password = '!20AMI_TS_ES18?'

file = msoffcrypto.OfficeFile(open(file_path, 'rb'))
file.load_key(password=password)

decrypted_stream = io.BytesIO()
file.decrypt(decrypted_stream)

decrypted_stream.seek(0)
Misogyny_test = pd.read_excel(decrypted_stream, engine='openpyxl')

print(Misogyny_test.head)

# %%
file_path = '/home/u161198/new_env/Datasets/SBW-vectors-300-min5.bin.gz'
W2Vembeddings = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
text_column = Misogyny_train['tweet']
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
Misogyny_train['embeddings'] = all_embeddings

# averaging the word embeddings to get the sentence embeddings
Misogyny_train['sentence_embedding'] = Misogyny_train['embeddings'].apply(lambda x: np.mean(x, axis=0))
print(Misogyny_train['sentence_embedding'].head())

# %%
text_column = Misogyny_test['tweet']
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
Misogyny_test['embeddings'] = all_embeddings

# averaging the word embeddings to get the sentence embeddings
Misogyny_test['sentence_embedding'] = Misogyny_test['embeddings'].apply(lambda x: np.mean(x, axis=0))
print(Misogyny_test['sentence_embedding'].head())

# %%
# SVM Classifer
X_test = np.array(Misogyny_test['embeddings'].tolist())
y_test = Misogyny_test['misogynous'].values

X_train = np.array(Misogyny_train['embeddings'].tolist())  # Convert list of lists to numpy array
y_train = Misogyny_train['misogynous'].values

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
precision = precision_score(y_test, predictions, average='binary', pos_label=1)  # Adjust according to your labels
recall = recall_score(y_test, predictions, average='binary', pos_label=1)        # Adjust according to your labels
f1 = f1_score(y_test, predictions, average='binary', pos_label=1)               # Adjust according to your labels

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")



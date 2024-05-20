# %%

import numpy as np
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV


file_path = '/home/u161198/new_env/Datasets/Racism/data/input_tweets.csv'
Racism = pd.read_csv(file_path, delimiter=";")


# retrieving the SBW embeddings for words train dataset
file_path = '/home/u161198/new_env/Datasets/SBW-vectors-300-min5.bin.gz'
W2Vembeddings = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
text_column = Racism['text']
sentence_embeddings = []
for text in text_column:
  words = text.split()
  row_embeddings = [W2Vembeddings[word] for word in words if word in W2Vembeddings]
  if row_embeddings:
        sentence_embedding = np.mean(row_embeddings, axis=0)
  else:
        sentence_embedding = np.zeros(W2Vembeddings.vector_size)

  sentence_embeddings.append(sentence_embedding)
Racism['sentence_embedding']= list(sentence_embeddings)

print(Racism[['text','sentence_embedding']].head())

# train test split

X = np.array(Racism['sentence_embedding'].tolist())  # This contains all columns except the label
y = Racism['target'].values  # This is the target label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM with a pipeline that includes scaling
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Fit the model
clf.fit(X_train, y_train)

# Predict and evaluate
predictions = clf.predict(X_test)

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



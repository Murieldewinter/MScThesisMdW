
import torch
from transformers import XGLMModel, XGLMTokenizer
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# reading the training dataset
file_path = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/EXIST_2021_Dataset/training/EXIST2021_training.tsv'

# Read the .tsv file
Sexism_complete_train = pd.read_csv(file_path, delimiter='\t')

Sexism_train= Sexism_complete_train[(Sexism_complete_train['language']=='es')& (Sexism_complete_train['source']=='twitter')]
texts_es = Sexism_train['text'].tolist()
for text in texts_es[:20]:
  print(text)
file_path_es = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/EXIST_2021_Dataset/training/texts_es.tsv'

# Read the testing dataset
file_path = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/EXIST_2021_Dataset/test/EXIST2021_test_labeled.tsv'
Sexism_complete_test = pd.read_csv(file_path, delimiter='\t')

Sexism_test= Sexism_complete_test[(Sexism_complete_test['language']=='es')& (Sexism_complete_test['source']=='twitter')]
texts_es = Sexism_test['text'].tolist()
for text in texts_es[:20]:
  print(text)
file_path_es = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/EXIST_2021_Dataset/test/texts_es.tsv'

# Load the model and tokenizer
model = XGLMModel.from_pretrained('facebook/xglm-7.5B')
tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-7.5B')

# Example dataset
texts = Sexism_train['text']

# Prepare model
model.eval()

# Store embeddings
embeddings = []

# Process each text
for text in texts:
    # Tokenize and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings
    last_hidden_states = outputs.last_hidden_state
    sentence_embeddings = last_hidden_states.mean(dim=1)

    # Store embeddings
    embeddings.append(sentence_embeddings)

# Print embeddings
for embedding in embeddings:
    print(embedding)

import numpy as np

# Convert tensor embeddings to lists and store them in the DataFrame
Sexism_train['embeddings'] = [embedding.cpu().numpy().tolist()[0] for embedding in embeddings]

# Check the DataFrame to see the new embeddings column
print(Sexism_train.head())

# Load the model and tokenizer
model = XGLMModel.from_pretrained('facebook/xglm-7.5B')
tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-7.5B')

# Example dataset
texts = Sexism_test['text']

# Prepare model
model.eval()

# Store embeddings
embeddings = []

# Process each text
for text in texts:
    # Tokenize and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings
    last_hidden_states = outputs.last_hidden_state
    sentence_embeddings = last_hidden_states.mean(dim=1)

    # Store embeddings
    embeddings.append(sentence_embeddings)

# Print embeddings
for embedding in embeddings:
    print(embedding)

Sexism_test_test['embeddings'] = [embedding.cpu().numpy().tolist()[0] for embedding in embeddings]

# Check the DataFrame to see the new embeddings column
print(Sexism_test.head())

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

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

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
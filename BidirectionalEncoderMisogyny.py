import torch
import numpy as np
import pandas as pd
import msoffcrypto
import io
from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences

file_path = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/es_AMI_TrainingSet_NEW (2).xlsx'
password = '!20AMI_ES18?'

file = msoffcrypto.OfficeFile(open(file_path, 'rb'))
file.load_key(password=password)

decrypted_stream = io.BytesIO()
file.decrypt(decrypted_stream)

decrypted_stream.seek(0)
Misogyny_train = pd.read_excel(decrypted_stream, engine='openpyxl')

print(Misogyny_train.head)

file_path = '/Users/MurieldeWinter/anaconda3/envs/new_env/Datasets/es_AMI_TestSet.xlsx'
password = '!20AMI_TS_ES18?'

file = msoffcrypto.OfficeFile(open(file_path, 'rb'))
file.load_key(password=password)

decrypted_stream = io.BytesIO()
file.decrypt(decrypted_stream)

decrypted_stream.seek(0)
Misogyny_test = pd.read_excel(decrypted_stream, engine='openpyxl')

print(Misogyny_test.head)

tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = model(**encoded_input)
last_hidden_states= outputs.last_hidden_state
print(last_hidden_states.shape)

BETO = 'dccuchile/bert-base-spanish-wwm-cased'
text_column = Misogyny_train['text']
tokenizer = BertTokenizer.from_pretrained(BETO)
# load
model = BertModel.from_pretrained(BETO)

# Assuming text_column is a pandas Series object containing text
input_text_list = text_column.tolist()

# tokenizer-> token_id
input_ids_list = [tokenizer.encode(text, add_special_tokens=True) for text in input_text_list]

# Pad sequences to ensure they all have the same length
max_length = max(len(ids) for ids in input_ids_list)
input_ids_list = pad_sequences(input_ids_list, maxlen=max_length, dtype="long", value=0, truncating="post", padding="post")

# Create attention mask
attention_mask = [[int(token_id > 0) for token_id in input_ids] for input_ids in input_ids_list]

# Convert to torch tensors
input_ids_tensor = torch.tensor(input_ids_list)
attention_mask_tensor = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids_tensor, attention_mask=attention_mask_tensor)[0]  # Models outputs are now tuples
last_hidden_states = last_hidden_states.mean(1)
print(last_hidden_states)

# Convert the tensor of embeddings to a list of lists
embeddings_list = last_hidden_states.cpu().detach().numpy().tolist()

# Ensure that the length of embeddings_list matches the number of rows in smalldataset
assert len(embeddings_list) == len(Misogyny_train), "The number of embeddings must match the number of rows in the dataset."

# Add the embeddings as a new column to the DataFrame
Misogyny_train['embeddings'] = embeddings_list

# Check the DataFrame to see the new 'embeddings' column
print(Misogyny_train.head())

BETO = 'dccuchile/bert-base-spanish-wwm-cased'
text_column = Misogyny_test['text']
tokenizer = BertTokenizer.from_pretrained(BETO)
# load
model = BertModel.from_pretrained(BETO)

# Assuming text_column is a pandas Series object containing text
input_text_list = text_column.tolist()

# tokenizer-> token_id
input_ids_list = [tokenizer.encode(text, add_special_tokens=True) for text in input_text_list]

# Pad sequences to ensure they all have the same length
max_length = max(len(ids) for ids in input_ids_list)
input_ids_list = pad_sequences(input_ids_list, maxlen=max_length, dtype="long", value=0, truncating="post", padding="post")

# Create attention mask
attention_mask = [[int(token_id > 0) for token_id in input_ids] for input_ids in input_ids_list]

# Convert to torch tensors
input_ids_tensor = torch.tensor(input_ids_list)
attention_mask_tensor = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids_tensor, attention_mask=attention_mask_tensor)[0]  # Models outputs are now tuples
last_hidden_states = last_hidden_states.mean(1)
print(last_hidden_states)

# Convert the tensor of embeddings to a list of lists
embeddings_list = last_hidden_states.cpu().detach().numpy().tolist()

# Ensure that the length of embeddings_list matches the number of rows in smalldataset
assert len(embeddings_list) == len(Misogyny_test), "The number of embeddings must match the number of rows in the dataset."

# Add the embeddings as a new column to the DataFrame
Misogyny_test['embeddings'] = embeddings_list

# Check the DataFrame to see the new 'embeddings' column
print(Misogyny_test.head())

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# SVM Classifer
X_test = np.array(Misogyny_test['embeddings'].tolist())
y_test = Misogyny_test['task1'].values

X_train = np.array(Misogyny_train['embeddings'].tolist())  # Convert list of lists to numpy array
y_train = Misogyny_train['task1'].values

# Create an SVM with a pipeline that includes scaling
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Fit the model
clf.fit(X_train, y_train)

# Predict and evaluate
predictions = clf.predict(X_test)

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
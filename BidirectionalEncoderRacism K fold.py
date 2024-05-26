

# %%
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# %%
# reading the dataset
file_path = '/home/u161198/new_env/Datasets/Racism/data/input_tweets.csv'
Racism = pd.read_csv(file_path, delimiter=";")


# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# BETO base uncased
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

# Loop through each fold
for train_index, test_index in kf.split(Racism):
    train, test = Racism.iloc[train_index], Racism.iloc[test_index]
    
    # making strings
    list_of_strings = train['text'].astype(str).tolist()
    text = list_of_strings
    
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
    
    # BETO training embeddings
    BETO = 'dccuchile/bert-base-spanish-wwm-cased'
    text_column = train['text']
    tokenizer = BertTokenizer.from_pretrained(BETO)
    model = BertModel.from_pretrained(BETO)
    
    input_text_list = text_column.tolist()
    
    input_ids_list = [tokenizer.encode(text, add_special_tokens=True) for text in input_text_list]
    input_ids_list = [torch.tensor(ids) for ids in input_ids_list]
    
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    
    attention_mask = [[int(token_id > 0) for token_id in input_ids] for input_ids in input_ids_list]
    
    input_ids_tensor = torch.tensor(input_ids_list)
    attention_mask_tensor = torch.tensor(attention_mask)
    
    with torch.no_grad():
        last_hidden_states = model(input_ids_tensor, attention_mask=attention_mask_tensor)[0]


# Collect results for each fold
confusion_matrices = []
classification_reports = []

for train_index, test_index in kf.split(X):
    train_texts, test_texts = [X[i] for i in train_index], [X[i] for i in test_index]
    train_labels, test_labels = [y[i] for i in train_index], [y[i] for i in test_index]
    
    # Tokenize and encode sequences for training set
    train_encoded_input = tokenizer(train_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        train_outputs = model(**train_encoded_input)
    train_last_hidden_states = train_outputs.last_hidden_state
    
    # Tokenize and encode sequences for test set
    test_encoded_input = tokenizer(test_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        test_outputs = model(**test_encoded_input)
    test_last_hidden_states = test_outputs.last_hidden_state
    
    # Flatten the embeddings for SVM input
    train_features = train_last_hidden_states[:, 0, :].numpy()
    test_features = test_last_hidden_states[:, 0, :].numpy()
    
    # Train the SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(train_features, train_labels)
    
    # Make predictions
    test_predictions = svm_classifier.predict(test_features)
    
    # Evaluate the predictions
    cm = confusion_matrix(test_labels, test_predictions)
    cr = classification_report(test_labels, test_predictions, output_dict=True)
    
    confusion_matrices.append(cm)
    classification_reports.append(cr)

# Print results for each fold
for i in range(len(confusion_matrices)):
    print(f"Fold {i+1}")
    print("Confusion Matrix:")
    print(confusion_matrices[i])
    print("Classification Report:")
    print(pd.DataFrame(classification_reports[i]).transpose())

# Convert classification reports to DataFrames and compute average
report_dfs = [pd.DataFrame(report).transpose() for report in classification_reports]
avg_classification_report = sum(report_dfs) / len(report_dfs)

print("Average Classification Report:")
print(avg_classification_report)

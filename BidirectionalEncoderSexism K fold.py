import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertModel, BertTokenizer

# Load the datasets
file_path_train = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/training/EXIST2021_training.tsv'
file_path_test = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/test/EXIST2021_test_labeled.tsv'

Sexism_complete_train = pd.read_csv(file_path_train, delimiter='\t')
Sexism_complete_test = pd.read_csv(file_path_test, delimiter='\t')

# Filter datasets
Sexism_train = Sexism_complete_train[(Sexism_complete_train['language'] == 'es') & (Sexism_complete_train['source'] == 'twitter')]
Sexism_test = Sexism_complete_test[(Sexism_complete_test['language'] == 'es') & (Sexism_complete_test['source'] == 'twitter')]

# Merge datasets
Sexism_data = pd.concat([Sexism_train, Sexism_test])

# BETO model and tokenizer
BETO = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(BETO)
model = BertModel.from_pretrained(BETO)

# Feature extraction function using BETO
def extract_features(text_list):
    input_ids_list = [tokenizer.encode(text, add_special_tokens=True) for text in text_list]
    input_ids_list = [torch.tensor(ids) for ids in input_ids_list]
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask = [[int(token_id > 0) for token_id in input_ids] for input_ids in input_ids_list]

    input_ids_tensor = torch.tensor(input_ids_list)
    attention_mask_tensor = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids_tensor, attention_mask=attention_mask_tensor)[0]
    return last_hidden_states.mean(1).cpu().numpy()

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
texts = Sexism_data['text'].tolist()
labels = Sexism_data['label'].tolist()  # Assuming 'label' column exists

all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for train_index, test_index in kf.split(texts):
    X_train, X_test = [texts[i] for i in train_index], [texts[i] for i in test_index]
    y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]

    # Extract features using BETO
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # Train SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train_features, y_train)

    # Predict and evaluate
    y_pred = svm.predict(X_test_features)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    all_metrics['accuracy'].append(accuracy)
    all_metrics['precision'].append(precision)
    all_metrics['recall'].append(recall)
    all_metrics['f1'].append(f1)
    
    print(f'Fold Accuracy: {accuracy}')
    print(f'Fold Precision: {precision}')
    print(f'Fold Recall: {recall}')
    print(f'Fold F1: {f1}')

# Calculate and print average metrics
avg_accuracy = np.mean(all_metrics['accuracy'])
avg_precision = np.mean(all_metrics['precision'])
avg_recall = np.mean(all_metrics['recall'])
avg_f1 = np.mean(all_metrics['f1'])

print(f'Average Accuracy: {avg_accuracy}')
print(f'Average Precision: {avg_precision}')
print(f'Average Recall: {avg_recall}')
print(f'Average F1: {avg_f1}')


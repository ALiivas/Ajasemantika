#!/usr/bin/env python
# coding: utf-8

# --------------------------------------------------
# -- Imports

#!pip install accelerate -U
#!pip install numpy==1.26.4
#!pip install optuna
#!pip install nervaluate

import os
os.environ['WANDB_DISABLED'] = 'true'
import json
from estnltk.converters import json_to_text
import sklearn
import numpy as np

from datasets import Dataset
import tensorflow as tf
import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
from nervaluate import Evaluator


# ------------------------------------------------
# -- Importing TimeMLCorpus
timeml_train_path = 'model_data/TimeML/train'
timeml_dev_path = 'model_data/TimeML/dev'
timeml_test_path = 'model_data/TimeML/test'

train_texts = []
dev_texts = []
test_texts = []

for filename in os.listdir(timeml_train_path):
    text_obj = json_to_text(file=os.path.join(timeml_train_path, filename))
    train_texts.append(text_obj)
    
for filename in os.listdir(timeml_dev_path):
    text_obj = json_to_text(file=os.path.join(timeml_dev_path, filename))
    dev_texts.append(text_obj)
    
for filename in os.listdir(timeml_test_path):
    text_obj = json_to_text(file=os.path.join(timeml_test_path, filename))
    test_texts.append(text_obj)
 
    
# -- Method for splitting corpus data into sentences with corresponding labels and IDs
def split_data(text_objects, layer):
    sentences = []
    labels = []
    
    for text in text_objects:
        for sentence in text.sentences:
            sent = []
            sent_labels = []
            for word in sentence.words:
                for w in text[layer]:
                    if word == w[0]:
                        sent.append(w[0].text)
                        sent_labels.append(w.nertag)
            sentences.append(sent)
            labels.append(sent_labels)
    
    assert len(sentences) == len(labels), "Different number of sentences and corresponding labels"
       
    return sentences, labels


# -- Creating train, dev and test data with labels
X_train, y_train = split_data(train_texts, 'gold_word_events_main')
X_dev, y_dev = split_data(dev_texts, 'gold_word_events_main')
X_test, y_test = split_data(test_texts, 'gold_word_events_main')


# -- Creating label list from all unique labels
train_labels_unique = list(set([l for label in y_train for l in label]))
dev_labels_unique = list(set([l for label in y_dev for l in label]))
test_labels_unique = list(set([l for label in y_test for l in label]))

label_list = list(set(train_labels_unique+dev_labels_unique+test_labels_unique))
label_list.append('PAD')


# -- Mapping labels to indexes
label2idx = {label: idx for idx, label in enumerate(label_list)}

# -- Mapping indexes to labels
idx2label = {idx: label for idx, label in enumerate(label_list)}


# -- Creating categorical labels
y_train_idx = [[label2idx.get(l) for l in label] for label in y_train]
y_dev_idx = [[label2idx.get(l) for l in label] for label in y_dev]
y_test_idx = [[label2idx.get(l) for l in label] for label in y_test]


# -- Method for creating datasets from sentences and categorical labels
def create_dataset(X, y):
    data = {'id': [], 'raw_labels': [], 'tokens': []}
    
    for i, sent in enumerate(X):
        data['id'].append(i)
        data['raw_labels'].append(y[i])
        data['tokens'].append(sent)
        
    return Dataset.from_dict(data)


# -- Creating train, dev and test datasets
train_dataset = create_dataset(X_train, y_train_idx)
dev_dataset = create_dataset(X_dev, y_dev_idx)
test_dataset = create_dataset(X_test, y_test_idx)


# -------------------------------------------------
# -- Tokenizing train, dev and test sets

# based on HuggingFace tutorial (https://huggingface.co/docs/transformers/tasks/token_classification)
def tokenize_and_align_labels(dataset):
    tokenized_inputs = tokenizer(dataset['tokens'], truncation=True, padding=True, max_length=75, is_split_into_words=True)
    
    aligned_labels = []
    
    for i, label in enumerate(dataset['raw_labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx       
        aligned_labels.append(label_ids)
        
    tokenized_inputs['labels'] = aligned_labels
    
    return tokenized_inputs


# -- Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("tartuNLP/EstBERT", do_lower_case=False)

# -- Tokenizing trainset, devset and testset
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, batch_size=2000)
tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched=True, batch_size=2000)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True, batch_size=2000)


# ------------------------------------------------
# -- Evaluation metrics

#seqeval = evaluate.load("seqeval")

# based on HuggingFace tutorial (https://huggingface.co/docs/transformers/tasks/token_classification)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    eval_nervaluate = Evaluator(true_labels, true_predictions, tags=['EVENT'], loader='list')
    eval_results, eval_results_by_tag, eval_result_indices, eval_result_indices_by_tag = eval_nervaluate.evaluate()
    
    return {
        "precision": eval_results['strict']['precision'],
        "recall": eval_results['strict']['recall'],
        "f1": eval_results['strict']['f1']
    }
    
    #results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    #return {
    #    "precision": results["overall_precision"],
    #    "recall": results["overall_recall"],
    #    "f1": results["overall_f1"],
    #    "accuracy": results["overall_accuracy"],
    #}
    

# -----------------------------------------------
# -- Model training and evaluation

device = 'cuda' if cuda.is_available() else 'cpu'

# parameter optimization
#def optuna_hp_space(trial):
#    return {
#        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
#        'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
#    }
    
#def model_init(trial):
#    model = AutoModelForTokenClassification.from_pretrained("tartuNLP/EstBERT", num_labels=4, id2label=idx2label, label2id=label2idx)
#    return model.to(device)
    #return model

GRID_SEARCH_PARAM = {
        "per_device_train_batch_size": [8, 16, 32],
        "learning_rate": [1e-5, 3e-5, 5e-5],
        "max_epochs":[3]
    }

def perform_param_grid_search():
    trial_model = AutoModelForTokenClassification.from_pretrained("tartuNLP/EstBERT", num_labels=len(label_list), id2label=idx2label, label2id=label2idx)
    trial_model.to(device)

    def trial_training(num_train_epochs, per_device_train_batch_size, learning_rate):
        trial_training_args = TrainingArguments(
            output_dir='estbert_events_main_trial_best',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            logging_dir='estbert_events_main_trial_best',
            logging_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1, #et jääks alles vaid parim ja viimane checkpoint
            learning_rate=learning_rate,
            eval_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model = 'f1'
            )
    
        trial_trainer = Trainer(
            model=trial_model,
            args=trial_training_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev
        )
        
        trial_trainer.train()
        res = trial_trainer.evaluate()
        return res['eval_f1']

    # Performing grid search
    best_trial_f1 = 0
    best_params = {'per_device_train_batch_size': None, 'learning_rate': None}

    for epoch in GRID_SEARCH_PARAM['max_epochs']:
        for i in range(len(GRID_SEARCH_PARAM['per_device_train_batch_size'])):
            for j in range(len(GRID_SEARCH_PARAM['learning_rate'])):
                trial_f1 = trial_training(epoch, GRID_SEARCH_PARAM['per_device_train_batch_size'][i], GRID_SEARCH_PARAM['learning_rate'][j])
                if trial_f1 > best_trial_f1:
                    print(f"Trial f1 score: {trial_f1}, best batch size: {best_params['per_device_train_batch_size']}, best learning rate: {best_params['learning_rate']}")
                    best_params['per_device_train_batch_size'] = GRID_SEARCH_PARAM['per_device_train_batch_size'][i]
                    best_params['learning_rate'] = GRID_SEARCH_PARAM['learning_rate'][j]
                    best_trial_f1 = trial_f1
                    
    return best_params
    
# Full model initializing and training

best_parameters = perform_param_grid_search()

# Saving best parameters to JSON
with open('estbert_events_main_best_best_params.json', 'w') as f:
    json.dump(best_parameters, f)
    
model = AutoModelForTokenClassification.from_pretrained("tartuNLP/EstBERT", num_labels=len(label_list), id2label=idx2label, label2id=label2idx)
model.to(device)

training_args = TrainingArguments(
    output_dir='estbert_events_main_param_tuning_best',
    num_train_epochs=40,
    per_device_train_batch_size=best_parameters['per_device_train_batch_size'],
    logging_dir='estbert_events_main_param_tuning_best',
    logging_strategy='epoch',
    save_strategy='epoch',
    learning_rate=best_parameters['learning_rate'],
    eval_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model = 'f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

#best_trial = trainer.hyperparameter_search(
#    direction='maximize',
#    backend='optuna',
#    hp_space=optuna_hp_space,
#    n_trials=20,
#)
trainer.train()

training_results = trainer.evaluate()

# -- Saving evaluation results to JSON
with open('estbert_events_main_param_tuning_best_eval.json', 'w') as f:
    json.dump(training_results, f)

# -- Saving model
trainer.save_model("./estbert_events_main_param_tuning_best/best/")

# -- Evaluating on final test set with nervaluate
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=2)

final_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, predictions.label_ids)
    ]

final_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, predictions.label_ids)
    ]

nervaluate = Evaluator(final_labels, final_predictions, tags=['EVENT'], loader='list')
results, results_by_tag, result_indices, result_indices_by_tag = nervaluate.evaluate()

# -- Saving final test results to JSON
with open('estbert_events_main_param_tuning_best_final_test.json', 'w') as f:
    json.dump([results, results_by_tag, result_indices, result_indices_by_tag], f)


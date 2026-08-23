#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from estnltk import Text, Layer
from estnltk.converters import text_to_json, json_to_text, json_to_layer

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

     
def read_articles(corpus_embed_path):
    # train-test failinimede saamine
    fn_path = "model_data/TimeML/"

    train_files = []
    test_files = []

    for filename in os.listdir(fn_path+"train/"):
        train_files.append(json_to_text(file=os.path.join(fn_path+"train/", filename)))
    for filename in os.listdir(fn_path+"dev/"): # liidame dev hulga train hulgaga kokku
        train_files.append(json_to_text(file=os.path.join(fn_path+"dev/", filename)))
    for filename in os.listdir(fn_path+"test/"):
        test_files.append(json_to_text(file=os.path.join(fn_path+"test/", filename)))
    
    # embeddingutega lugemine train-dev-test hulkadesse   
    train_files_concat = []
    test_files_concat = []
    train_files_add = []
    test_files_add = []
    
    for text in train_files:
        filename = text.meta['filename']
        train_files_concat.append(json_to_layer(text, file=f"{corpus_embed_path}{filename}_embed_concat.json"))
        train_files_add.append(json_to_layer(text, file=f"{corpus_embed_path}{filename}_embed_add.json"))
    for text in test_files:
        filename = text.meta['filename']
        test_files_concat.append(json_to_layer(text, file=f"{corpus_embed_path}{filename}_embed_concat.json"))
        test_files_add.append(json_to_layer(text, file=f"{corpus_embed_path}{filename}_embed_add.json"))
    
    print("Lugesin sisse", len(train_files_concat), len(train_files_add), "treeningandmete faili.")
    print("Lugesin sisse", len(test_files_concat), len(test_files_add), "testandmete faili.")
    
    return train_files, test_files, train_files_concat, train_files_add, test_files_concat, test_files_add


def get_event_main_embeddings(text_list, text_embed_list, layer_type, aggregate):
    """
    Leiab ja tagastab sündmuste-ajaväljendite peasõnade embeddingud ning TLINK labelid.
    """
    agg_dct = {'SIMULTANEOUS': 'OVERLAP',
                  'INCLUDES': 'OVERLAP',
                  'IS_INCLUDED': 'OVERLAP',
                  'BEFORE': 'BEFORE',
                  'AFTER': 'AFTER',
                  'VAGUE': 'VAGUE'}
    embed = []
    tlink_labels = []
    for text_idx, text in enumerate(text_list):
        # sündmusfraasi peasõna vektor
        for idx1, tlink in enumerate(text["event_dct_tlinks"]):
            ev_word_spans = [text.words.get(span) for span in tlink.a_text.base_span]
            
            for idx2, word in enumerate(text.gold_word_events_main):
                if word.nertag == 'B-EVENT' or word.nertag == 'I-EVENT':
                    if text.words.get(word[0]) in ev_word_spans:
                        if layer_type:
                            # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                            if layer_type == "penultimate":
                                embed.append(np.asarray(text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                            # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                            elif layer_type == "last":
                                embed.append(np.asarray(text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                        else:
                            embed.append(np.asarray(text_embed_list[text_idx][idx2].bert_embedding))
            if aggregate:
                tlink_labels.append(agg_dct[tlink["rel_type"]])
            else:
                tlink_labels.append(tlink["rel_type"])
                
    assert len(tlink_labels) == len(embed), "different list lengths"
    return embed, tlink_labels

        
def get_event_mean_embeddings(text_list, text_embed_list, layer_type, aggregate):
    """
    Leiab ja tagastab sündmusfraaside sõnade embeddingute aritmeetilised keskmised väärtused ning TLINK labelid.
    """
    agg_dct = {'SIMULTANEOUS': 'OVERLAP',
                    'INCLUDES': 'OVERLAP',
                    'IS_INCLUDED': 'OVERLAP',
                    'BEFORE': 'BEFORE',
                    'AFTER': 'AFTER',
                    'VAGUE': 'VAGUE'}
    embed = []
    tlink_labels = []
    for text_idx, text in enumerate(text_list):
        # algsest sündmuste kihist saame kätte ajaliste kestuste labelid
        for idx1, tlink in enumerate(text["event_dct_tlinks"]):
            ev_word_spans = [text.words.get(span) for span in tlink.a_text.base_span]
        
            # leiame sündmusfraasi sõnade vektorid
            event_phrase_embed = []
            for idx2, word in enumerate(text.words):
                if word in ev_word_spans:
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            event_phrase_embed.append(text_embed_list[text_idx][idx2].bert_embedding[-1536:-768])
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            event_phrase_embed.append(text_embed_list[text_idx][idx2].bert_embedding[-768:])
                    else:
                        event_phrase_embed.append(text_embed_list[text_idx][idx2].bert_embedding)
            embed.append(np.mean(event_phrase_embed, 0))
            if aggregate:
                tlink_labels.append(agg_dct[tlink["rel_type"]])
            else:
                tlink_labels.append(tlink["rel_type"])                   
    assert len(tlink_labels) == len(embed), "different list lengths"
    return embed, tlink_labels


def run_pipeline(clf, model_save_path, results_save_path, clf_name, embed_type, X_train, y_train, X_test, y_test):
    """
    Treenib mudeli ja leiab õigsuse lõplikul testandmestikul.
    """
    # treenime mudeli
    clf.fit(X_train, y_train)
    if clf_name != "dummy":
        # salvestame mudeli faili
        pickle.dump(clf, open(f'{model_save_path}/{clf_name}_{embed_type}.pkl', 'wb'))
    # ennustame testandmestiku labelid
    y_pred = clf.predict(X_test)
    # leiame õigsuse
    acc_score = accuracy_score(y_test, y_pred)
    clsf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    # salvestame tulemused
    clsf_report.to_csv(f'{results_save_path}/{clf_name}_{embed_type}_clsf_report.csv', index= True)
    with open(f'{results_save_path}/ev_dct_accuracy.txt', 'a') as f:
        f.write(f'{clf_name}, {embed_type} accuracy: {acc_score}\n')


def get_all_embeddings(train_texts, test_texts, train_concat, test_concat, train_add, test_add, event_words, aggregate):

    event_embeddings = {
        'train_concat': [], # viimase nelja kihi konkatenatsioon
        'test_concat': [],
        'train_add': [], # viimase nelja kihi summa
        'test_add': [],
        'train_penult': [], # eelviimane kiht
        'test_penult': [],
        'train_last': [], # viimane kiht
        'test_last': []
    }

    # Kui on sündmusfraaside peasõnad
    if event_words == "main":
        event_embeddings['train_concat'] = [get_event_main_embeddings(train_texts, train_concat, None, aggregate)]
        event_embeddings['test_concat'] = [get_event_main_embeddings(test_texts, test_concat, None, aggregate)]
        event_embeddings['train_add'] = [get_event_main_embeddings(train_texts, train_add, None, aggregate)]
        event_embeddings['test_add'] = [get_event_main_embeddings(test_texts, test_add, None, aggregate)]
        event_embeddings['train_penult'] = [get_event_main_embeddings(train_texts, train_concat, 'penultimate', aggregate)]
        event_embeddings['test_penult'] = [get_event_main_embeddings(test_texts, test_concat, 'penultimate', aggregate)]
        event_embeddings['train_last'] = [get_event_main_embeddings(train_texts, train_concat, 'last', aggregate)]
        event_embeddings['test_last'] = [get_event_main_embeddings(test_texts, test_concat, 'last', aggregate)]
    # Kui on sündmusfraaside kõik sõnad
    elif event_words == "all":
        event_embeddings['train_concat'] = [get_event_mean_embeddings(train_texts, train_concat, None, aggregate)]
        event_embeddings['test_concat'] = [get_event_mean_embeddings(test_texts, test_concat, None, aggregate)]
        event_embeddings['train_add'] = [get_event_mean_embeddings(train_texts, train_add, None, aggregate)]
        event_embeddings['test_add'] = [get_event_mean_embeddings(test_texts, test_add, None, aggregate)]
        event_embeddings['train_penult'] = [get_event_mean_embeddings(train_texts, train_concat, 'penultimate', aggregate)]
        event_embeddings['test_penult'] = [get_event_mean_embeddings(test_texts, test_concat, 'penultimate', aggregate)]
        event_embeddings['train_last'] = [get_event_mean_embeddings(train_texts, train_concat, 'last', aggregate)]
        event_embeddings['test_last'] = [get_event_mean_embeddings(test_texts, test_concat, 'last', aggregate)]
        
    return event_embeddings


# tekstid ja embeddingud        
train_texts, test_texts, train_concat, train_add, test_concat, test_add = read_articles("embeddings/RoBERTa_embed_EstTimeML/")


def train_models(corpus_embed_path, 
                 model_save_path,
                 results_save_path,
                 aggregate
                 ):

    # tekstid ja embeddingud        
    train_texts, test_texts, train_concat, train_add, test_concat, test_add = read_articles("embeddings/RoBERTa_embed_EstTimeML/")

    # sündmusfraaside peasõnade embeddingud
    event_main_embeddings = get_all_embeddings(train_texts, 
                                           test_texts, 
                                           train_concat, 
                                           test_concat, 
                                           train_add,
                                           test_add,
                                           event_words="main",
                                           aggregate=aggregate)

    # sündmusfraaside kõigi sõnade aritmeetilise keskmise embeddingud
    event_mean_embeddings = get_all_embeddings(train_texts, 
                                           test_texts, 
                                           train_concat, 
                                           test_concat, 
                                           train_add,
                                           test_add,
                                           event_words="all",
                                           aggregate=aggregate)

    # klasside arv treeningandmestikus
    n_classes = len(set(list(event_main_embeddings['train_concat'][1])))

    # klassifitseerijad
    svc_clf = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=0))
    rf_clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
    mlp_clf = make_pipeline(StandardScaler(), MLPClassifier(early_stopping=True, random_state=0))
    xgb_clf = make_pipeline(StandardScaler(), XGBClassifier(objective='multi:softmax', num_class=n_classes, random_state=0))
    dummy_clf = make_pipeline(StandardScaler(), DummyClassifier(strategy='most_frequent'))

    # label mappings
    labels2idx = {'SIMULTANEOUS': 0,
                'INCLUDES': 1,
                'IS_INCLUDED': 2,
                'BEFORE': 3,
                'AFTER': 4,
                'VAGUE': 5}
    if aggregate:
        labels2idx = {'OVERLAP': 0,
                    'BEFORE': 1,
                    'AFTER': 2,
                    'VAGUE': 3}

    # ---------treenimine ja tulemuste salvestamine-----------

    # nelja viimase kihi konkatenatsioon
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'main_concat', event_main_embeddings['train_concat'][0], event_main_embeddings['train_concat'][1], event_main_embeddings['test_concat'][0], event_main_embeddings['test_concat'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'main_concat', event_main_embeddings['train_concat'][0], event_main_embeddings['train_concat'][1], event_main_embeddings['test_concat'][0], event_main_embeddings['test_concat'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'main_concat', event_main_embeddings['train_concat'][0], event_main_embeddings['train_concat'][1], event_main_embeddings['test_concat'][0], event_main_embeddings['test_concat'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'main_concat', event_main_embeddings['train_concat'][0], [labels2idx[label] for label in event_main_embeddings['train_concat'][1]], event_main_embeddings['test_concat'][0], [labels2idx[label] for label in event_main_embeddings['test_concat'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'main_concat', event_main_embeddings['train_concat'][0], event_main_embeddings['train_concat'][1], event_main_embeddings['test_concat'][0], event_main_embeddings['test_concat'][1])

    # fraasi vektorite aritmeetiline keskmine
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'mean_concat', event_mean_embeddings['train_concat'][0], event_mean_embeddings['train_concat'][1], event_mean_embeddings['test_concat'][0], event_mean_embeddings['test_concat'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'mean_concat', event_mean_embeddings['train_concat'][0], event_mean_embeddings['train_concat'][1], event_mean_embeddings['test_concat'][0], event_mean_embeddings['test_concat'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'mean_concat', event_mean_embeddings['train_concat'][0], event_mean_embeddings['train_concat'][1], event_mean_embeddings['test_concat'][0], event_mean_embeddings['test_concat'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'mean_concat', event_mean_embeddings['train_concat'][0], [labels2idx[label] for label in event_mean_embeddings['train_concat'][1]], event_mean_embeddings['test_concat'][0], [labels2idx[label] for label in event_mean_embeddings['test_concat'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'mean_concat', event_mean_embeddings['train_concat'][0], event_mean_embeddings['train_concat'][1], event_mean_embeddings['test_concat'][0], event_mean_embeddings['test_concat'][1])

    # nelja viimase kihi summa
    # peasõna vektorid
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'main_add', event_main_embeddings['train_add'][0], event_main_embeddings['train_add'][1], event_main_embeddings['test_add'][0], event_main_embeddings['test_add'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'main_add', event_main_embeddings['train_add'][0], event_main_embeddings['train_add'][1], event_main_embeddings['test_add'][0], event_main_embeddings['test_add'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'main_add', event_main_embeddings['train_add'][0], event_main_embeddings['train_add'][1], event_main_embeddings['test_add'][0], event_main_embeddings['test_add'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'main_add', event_main_embeddings['train_add'][0], [labels2idx[label] for label in event_main_embeddings['train_add'][1]], event_main_embeddings['test_add'][0], [labels2idx[label] for label in event_main_embeddings['test_add'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'main_add', event_main_embeddings['train_add'][0], event_main_embeddings['train_add'][1], event_main_embeddings['test_add'][0], event_main_embeddings['test_add'][1])

    # fraasi vektorite aritmeetiline keskmine
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'mean_add', event_mean_embeddings['train_add'][0], event_mean_embeddings['train_add'][1], event_mean_embeddings['test_add'][0], event_mean_embeddings['test_add'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'mean_add', event_mean_embeddings['train_add'][0], event_mean_embeddings['train_add'][1], event_mean_embeddings['test_add'][0], event_mean_embeddings['test_add'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'mean_add', event_mean_embeddings['train_add'][0], event_mean_embeddings['train_add'][1], event_mean_embeddings['test_add'][0], event_mean_embeddings['test_add'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'mean_add', event_mean_embeddings['train_add'][0], [labels2idx[label] for label in event_mean_embeddings['train_add'][1]], event_mean_embeddings['test_add'][0], [labels2idx[label] for label in event_mean_embeddings['test_add'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'mean_add', event_mean_embeddings['train_add'][0], event_mean_embeddings['train_add'][1], event_mean_embeddings['test_add'][0], event_mean_embeddings['test_add'][1])

    # eelviimane kiht
    # peasõna vektorid
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'main_penult', event_main_embeddings['train_penult'][0], event_main_embeddings['train_penult'][1], event_main_embeddings['test_penult'][0], event_main_embeddings['test_penult'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'main_penult', event_main_embeddings['train_penult'][0], event_main_embeddings['train_penult'][1], event_main_embeddings['test_penult'][0], event_main_embeddings['test_penult'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'main_penult', event_main_embeddings['train_penult'][0], event_main_embeddings['train_penult'][1], event_main_embeddings['test_penult'][0], event_main_embeddings['test_penult'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'main_penult', event_main_embeddings['train_penult'][0], [labels2idx[label] for label in event_main_embeddings['train_penult'][1]], event_main_embeddings['test_penult'][0], [labels2idx[label] for label in event_main_embeddings['test_penult'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'main_penult', event_main_embeddings['train_penult'][0], event_main_embeddings['train_penult'][1], event_main_embeddings['test_penult'][0], event_main_embeddings['test_penult'][1])

    # fraasi vektorite aritmeetiline keskmine
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'mean_penult', event_mean_embeddings['train_penult'][0], event_mean_embeddings['train_penult'][1], event_mean_embeddings['test_penult'][0], event_mean_embeddings['test_penult'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'mean_penult', event_mean_embeddings['train_penult'][0], event_mean_embeddings['train_penult'][1], event_mean_embeddings['test_penult'][0], event_mean_embeddings['test_penult'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'mean_penult', event_mean_embeddings['train_penult'][0], event_mean_embeddings['train_penult'][1], event_mean_embeddings['test_penult'][0], event_mean_embeddings['test_penult'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'mean_penult', event_mean_embeddings['train_penult'][0], [labels2idx[label] for label in event_mean_embeddings['train_penult'][1]], event_mean_embeddings['test_penult'][0], [labels2idx[label] for label in event_mean_embeddings['test_penult'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'mean_penult', event_mean_embeddings['train_penult'][0], event_mean_embeddings['train_penult'][1], event_mean_embeddings['test_penult'][0], event_mean_embeddings['test_penult'][1])
    
    # viimane kiht
    # peasõna vektorid
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'main_last', event_main_embeddings['train_last'][0], event_main_embeddings['train_last'][1], event_main_embeddings['test_last'][0], event_main_embeddings['test_last'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'main_last', event_main_embeddings['train_last'][0], event_main_embeddings['train_last'][1], event_main_embeddings['test_last'][0], event_main_embeddings['test_last'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'main_last', event_main_embeddings['train_last'][0], event_main_embeddings['train_last'][1], event_main_embeddings['test_last'][0], event_main_embeddings['test_last'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'main_last', event_main_embeddings['train_last'][0], [labels2idx[label] for label in event_main_embeddings['train_last'][1]], event_main_embeddings['test_last'][0], [labels2idx[label] for label in event_main_embeddings['test_last'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'main_last', event_main_embeddings['train_last'][0], event_main_embeddings['train_last'][1], event_main_embeddings['test_last'][0], event_main_embeddings['test_last'][1])

    # fraasi vektorite aritmeetiline keskmine
    run_pipeline(svc_clf, model_save_path, results_save_path, 'SVC', 'mean_last', event_mean_embeddings['train_last'][0], event_mean_embeddings['train_last'][1], event_mean_embeddings['test_last'][0], event_mean_embeddings['test_last'][1])
    run_pipeline(rf_clf, model_save_path, results_save_path, 'RF', 'mean_last', event_mean_embeddings['train_last'][0], event_mean_embeddings['train_last'][1], event_mean_embeddings['test_last'][0], event_mean_embeddings['test_last'][1])
    run_pipeline(mlp_clf, model_save_path, results_save_path, 'MLP', 'mean_last', event_mean_embeddings['train_last'][0], event_mean_embeddings['train_last'][1], event_mean_embeddings['test_last'][0], event_mean_embeddings['test_last'][1])
    run_pipeline(xgb_clf, model_save_path, results_save_path, 'XGB', 'mean_last', event_mean_embeddings['train_last'][0], [labels2idx[label] for label in event_mean_embeddings['train_last'][1]], event_mean_embeddings['test_last'][0], [labels2idx[label] for label in event_mean_embeddings['test_last'][1]])
    run_pipeline(dummy_clf, model_save_path, results_save_path, 'dummy', 'mean_last', event_mean_embeddings['train_last'][0], event_mean_embeddings['train_last'][1], event_mean_embeddings['test_last'][0], event_mean_embeddings['test_last'][1])


###############################################3
# EXAMPLES OF IMPLEMENTATION
#
train_models(corpus_embed_path="embeddings/RoBERTa_embed_EstTimeML/", 
             model_save_path="estroberta_ev_dct_models", 
             results_save_path="estroberta_ev_dct_res",
             aggregate=False
             )

train_models(corpus_embed_path="embeddings/BERT_embed_EstTimeML/", 
             model_save_path="estbert_ev_dct_agg_models", 
             results_save_path="estroberta_ev_dct_agg_res",
             aggregate=True
             )

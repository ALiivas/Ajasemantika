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

     
def read_articles():
    # train-test failinimede saamine
    fn_path = "TempFact/"

    train_files = []
    test_files = []

    for filename in os.listdir(fn_path+"train_larger/"):
        train_files.append(json_to_text(file=os.path.join(fn_path+"train_larger/", filename)))
    for filename in os.listdir(fn_path+"test/"):
        test_files.append(json_to_text(file=os.path.join(fn_path+"test/", filename)))
    
    # embeddingutega lugemine train-dev-test hulkadesse    
    embed_path_estbert = "embeddings/RoBERTa_embed_temp_facts/"
    
    train_files_concat = []
    test_files_concat = []
    train_files_add = []
    test_files_add = []
    
    for text in train_files:
        filename = text.meta['filename']
        train_files_concat.append(json_to_layer(text, file=f"{embed_path_estbert}{filename}_embed_concat.json"))
        train_files_add.append(json_to_layer(text, file=f"{embed_path_estbert}{filename}_embed_add.json"))
    for text in test_files:
        filename = text.meta['filename']
        test_files_concat.append(json_to_layer(text, file=f"{embed_path_estbert}{filename}_embed_concat.json"))
        test_files_add.append(json_to_layer(text, file=f"{embed_path_estbert}{filename}_embed_add.json"))
    
    print("Lugesin sisse", len(train_files_concat), len(train_files_add), "treeningandmete faili.")
    print("Lugesin sisse", len(test_files_concat), len(test_files_add), "testandmete faili.")
    
    return train_files, test_files, train_files_concat, train_files_add, test_files_concat, test_files_add


def get_event_main_mean_embeddings(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmuste embeddingud ning tlink labelid.
    """
    agg_dct = {"instant": "less_than_day",
               "seconds": "less_than_day",
               "minutes": "less_than_day",
               "hours": "less_than_day",
               "days": "less_than_year",
               "weeks": "less_than_year",
               "months": "less_than_year",
               "years": "more_than_year",
               "centuries": "more_than_year",
               "forever": "more_than_year"}
    
    embed = []
    phrase_embed = []
    duration_labels = []
    phrase_lengths = []
    for text_idx, text in enumerate(text_list):
        #word_spans = []
        #ev_phrase_spans = []
        # sündmusfraasi peasõna vektor
        for idx, word in enumerate(text.gold_word_events_main):
            if word.nertag == 'B-EVENT' or word.nertag == 'I-EVENT':
                #has_corresponding_phrase = False
                word_span = text.words.get(word[0])
                event_phrase = None
                duration_label = None
                for idx2, event in enumerate(text.events):
                    for event_word in event:
                        if text.words.get(event_word) == word_span:
                            event_phrase = event
                            duration_label = event.duration
                            break
                    if event_phrase is not None and duration_label is not None:
                        break
            
                if word_span and event_phrase is not None and duration_label is not None:
                    #word_spans.append(text.words.get(word[0]))
                    emb = None
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            emb = np.asarray(text_embed_list[text_idx][idx].bert_embedding[-1536:-768])
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            emb = np.asarray(text_embed_list[text_idx][idx].bert_embedding[-768:])
                    else:
                        emb = np.asarray(text_embed_list[text_idx][idx].bert_embedding)
                    
                    event_phrase_spans = [text.words.get(wrd) for wrd in event_phrase]
                    event_phrase_embed = []
                    for idx2, word2 in enumerate(text.gold_word_events):
                        if text.words.get(word2[0]) in event_phrase_spans:
                            if layer_type:
                                # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                                if layer_type == "penultimate":
                                    event_phrase_embed.append(text_embed_list[text_idx][idx2].bert_embedding[-1536:-768])
                                # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                                elif layer_type == "last":
                                    event_phrase_embed.append(text_embed_list[text_idx][idx2].bert_embedding[-768:])
                            else:
                                event_phrase_embed.append(text_embed_list[text_idx][idx2].bert_embedding)
                    # leiame fraasi vektorite aritmeetilise keskmise
                    if len(event_phrase_embed) > 0:
                        phrase_lengths.append(len(event_phrase_embed))
                        embed.append(emb)
                        phrase_embed.append(np.mean(event_phrase_embed, 0))
                        duration_labels.append(agg_dct[duration_label])
    
    print("phrase lengths distribution:")
    from collections import Counter
    print(Counter(phrase_lengths))
                
    assert len(duration_labels) == len(embed) == len(phrase_embed), "different list lengths"
    return embed, phrase_embed, duration_labels


def run_pipeline(clf, save_path, clf_name, embed_type, X_train, y_train, X_test, y_test):
    """
    Treenib ja leiab mudeli õigsuse testandmestikul.
    """
    # treenime mudeli
    clf.fit(X_train, y_train)
    if clf_name != "dummy":
        # salvestame mudeli faili
        pickle.dump(clf, open(f'estroberta_ev_durations_agg_models/estroberta_{clf_name}_{embed_type}.pkl', 'wb'))
    # ennustame testandmestiku labelid
    y_pred = clf.predict(X_test)
    # leiame õigsuse
    acc_score = accuracy_score(y_test, y_pred)
    clsf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    # salvestame tulemused
    clsf_report.to_csv(f'{save_path}/{clf_name}_{embed_type}_clsf_report.csv', index= True)
    with open(f'{save_path}/estroberta_ev_dur_accuracy.txt', 'a') as f:
        f.write(f'{clf_name}, {embed_type} accuracy: {acc_score}\n')
    
# tekstid ja embeddingud        
train_texts, test_texts, train_concat, train_add, test_concat, test_add = read_articles()

# viimase nelja kihi konkatenatsioon
X_train_main_concat, X_train_mean_concat, y_train_concat = get_event_main_mean_embeddings(train_texts, train_concat, None)
X_test_main_concat, X_test_mean_concat, y_test_concat = get_event_main_mean_embeddings(test_texts, test_concat, None)

# viimase nelja kihi summa
X_train_main_add, X_train_mean_add, y_train_add = get_event_main_mean_embeddings(train_texts, train_add, None)
X_test_main_add, X_test_mean_add, y_test_add = get_event_main_mean_embeddings(test_texts, test_add, None)

# eelviimane kiht
X_train_main_penult, X_train_mean_penult, y_train_penult = get_event_main_mean_embeddings(train_texts, train_concat, 'penultimate')
X_test_main_penult, X_test_mean_penult, y_test_penult = get_event_main_mean_embeddings(test_texts, test_concat, 'penultimate')

# viimane kiht
X_train_main_last, X_train_mean_last, y_train_last = get_event_main_mean_embeddings(train_texts, train_concat, 'last')
X_test_main_last, X_test_mean_last, y_test_last = get_event_main_mean_embeddings(test_texts, test_concat, 'last')

# klasside arv treeningandmestikus
n_classes = len(set(list(y_train_concat)))

# klassifitseerijad
svc_clf = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=0))
rf_clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
mlp_clf = make_pipeline(StandardScaler(), MLPClassifier(early_stopping=True, random_state=0))
xgb_clf = make_pipeline(StandardScaler(), XGBClassifier(objective='multi:softmax', num_class=n_classes, random_state=0))
dummy_clf = make_pipeline(StandardScaler(), DummyClassifier(strategy='most_frequent'))

# LabelEncoder
#le = LabelEncoder()

labels2idx = {'less_than_day': 0,
              'less_than_year': 1,
              'more_than_year': 2}

# ---------treenimine ja tulemused-----------

# nelja viimase kihi konkatenatsioon
# peasõna vektorid
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'main_concat', X_train_main_concat, [labels2idx[label] for label in y_train_concat], X_test_main_concat, [labels2idx[label] for label in y_test_concat])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'mean_concat', X_train_mean_concat, [labels2idx[label] for label in y_train_concat], X_test_mean_concat, [labels2idx[label] for label in y_test_concat])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)

# nelja viimase kihi summa
# peasõna vektorid
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'main_add', X_train_main_add, [labels2idx[label] for label in y_train_add], X_test_main_add, [labels2idx[label] for label in y_test_add])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'mean_add', X_train_mean_add, [labels2idx[label] for label in y_train_add], X_test_mean_add, [labels2idx[label] for label in y_test_add])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)

# eelviimane kiht
# peasõna vektorid
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'main_penult', X_train_main_penult, [labels2idx[label] for label in y_train_penult], X_test_main_penult, [labels2idx[label] for label in y_test_penult])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'mean_penult', X_train_mean_penult, y_train_penult, X_test_mean_penult, y_test_penult)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'mean_penult', X_train_mean_penult, y_train_penult, X_test_mean_penult, y_test_penult)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'mean_penult', X_train_mean_penult, y_train_penult, X_test_mean_penult, y_test_penult)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'mean_penult', X_train_mean_penult, [labels2idx[label] for label in y_train_penult], X_test_mean_penult, [labels2idx[label] for label in y_test_penult])

# viimane kiht
# peasõna vektorid
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'main_last', X_train_main_last, [labels2idx[label] for label in y_train_last], X_test_main_last, [labels2idx[label] for label in y_test_last])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estroberta_ev_durations_agg_res', 'SVC', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)
run_pipeline(rf_clf, 'estroberta_ev_durations_agg_res', 'RF', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)
run_pipeline(mlp_clf, 'estroberta_ev_durations_agg_res', 'MLP', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)
run_pipeline(xgb_clf, 'estroberta_ev_durations_agg_res', 'XGB', 'mean_last', X_train_mean_last, [labels2idx[label] for label in y_train_last], X_test_mean_last, [labels2idx[label] for label in y_test_last])
run_pipeline(dummy_clf, 'estroberta_ev_durations_agg_res', 'dummy', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)

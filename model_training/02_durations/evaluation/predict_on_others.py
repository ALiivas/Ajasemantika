#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from estnltk import Text, Layer
from estnltk.converters import text_to_json, json_to_text, json_to_layer

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

def read_articles(data_dir: str, embed_path: str):
    # test failinimede saamine
    fn_path = "./masters_thesis/durations/TempFact/"
    #fn_path = "durations/TempFact/"

    test_files = []

    for filename in os.listdir(fn_path+data_dir):
        test_files.append(json_to_text(file=os.path.join(fn_path+data_dir, filename)))
    
    # embeddingutega lugemine testhulka, hetkel EstBERT
    #embed_path = "embeddings/BERT_embed_temp_facts/"
    
    test_files_concat = []
    test_files_add = []
    
    for text in test_files:
        filename = text.meta['filename']
        test_files_concat.append(json_to_layer(text, file=f"{embed_path}{filename}_embed_concat.json"))
        test_files_add.append(json_to_layer(text, file=f"{embed_path}{filename}_embed_add.json"))
    
    print("Lugesin sisse", len(test_files_concat), len(test_files_add), "testandmete faili.")
    
    return test_files, test_files_concat, test_files_add

def get_event_main_mean_embeddings(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmuste embeddingud ning tlink labelid.
    """
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
                        duration_labels.append(duration_label)
    
    print("phrase lengths distribution:")
    from collections import Counter
    print(Counter(phrase_lengths))
                
    assert len(duration_labels) == len(embed) == len(phrase_embed), "different list lengths"
    return embed, phrase_embed, duration_labels

def get_event_main_mean_embeddings_tri(text_list, text_embed_list, layer_type):
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

def get_event_main_mean_embeddings_bin(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmuste embeddingud ning tlink labelid.
    """
    agg_dct = {"instant": "less_than_day",
               "seconds": "less_than_day",
               "minutes": "less_than_day",
               "hours": "less_than_day",
               "days": "more_than_day",
               "weeks": "more_than_day",
               "months": "more_than_day",
               "years": "more_than_day",
               "centuries": "more_than_day",
               "forever": "more_than_day"}
    
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


def run_pipeline(clf, save_path, model_type, clf_name, embed_type, agg_type, X_test, y_test):
    """
    Leiab mudeli õigsuse testandmestikul.
    """
    # ennustame testandmestiku labelid
    y_pred = clf.predict(X_test)
    print(y_pred)
    # leiame õigsuse
    acc_score = accuracy_score(y_test, y_pred)
    # salvestame tulemused
    with open(f'{save_path}/rkogu_accuracy.txt', 'a') as f:
        f.write(f'{model_type}, {clf_name}, {embed_type}, {agg_type} accuracy: {acc_score}\n')
    
# tekstid ja embeddingud        
test_texts, test_concat, test_add = read_articles('rkogu/', 'masters_thesis/durations/embeddings/RoBERTa_embed_temp_facts/')

# viimased neli kihti liidetuna (konkatenatsioon)
X_test_main_concat, X_test_mean_concat, y_test_concat = get_event_main_mean_embeddings(test_texts, test_concat, None)
X_test_main_concat_tri, X_test_mean_concat_tri, y_test_concat_tri = get_event_main_mean_embeddings_tri(test_texts, test_concat, None)
X_test_main_concat_bin, X_test_mean_concat_bin, y_test_concat_bin = get_event_main_mean_embeddings_bin(test_texts, test_concat, None)

# eelviimane kiht
X_test_main_penult, X_test_mean_penult, y_test_penult = get_event_main_mean_embeddings(test_texts, test_concat, 'penultimate')
X_test_main_penult_tri, X_test_mean_penult_tri, y_test_penult_tri = get_event_main_mean_embeddings_tri(test_texts, test_concat, 'penultimate')
X_test_main_penult_bin, X_test_mean_penult_bin, y_test_penult_bin = get_event_main_mean_embeddings_bin(test_texts, test_concat, 'penultimate')

# viimane kiht
X_test_main_last, X_test_mean_last, y_test_last = get_event_main_mean_embeddings(test_texts, test_concat, 'last')
X_test_main_last_tri, X_test_mean_last_tri, y_test_last_tri = get_event_main_mean_embeddings_tri(test_texts, test_concat, 'last')
X_test_main_last_bin, X_test_mean_last_bin, y_test_last_bin = get_event_main_mean_embeddings_bin(test_texts, test_concat, 'last')

# LabelEncoder
#le = LabelEncoder()
labels2idx_3 = {'less_than_day': 0,
                'less_than_year': 1,
                'more_than_year': 2}

labels2idx_2 = {'less_than_day': 0,
                'more_than_day': 1}

clf1 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_models/estbert_RF_main_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf2 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_models/estbert_RF_mean_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf3 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_models/estroberta_RF_main_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf4 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_models/estroberta_RF_mean_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf5 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_agg_models/estbert_XGB_main_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf6 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_agg_models/estbert_XGB_mean_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf7 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_agg_models/estroberta_SVC_main_penult.pkl', 'rb')) # TEHTUD, TEHTUD
clf8 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_agg_models/estroberta_SVC_mean_penult.pkl', 'rb')) # TEHTUD, TEHTUD
clf9 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_agg_binary_models/estbert_XGB_main_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf10 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_agg_binary_models/estbert_XGB_mean_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf11 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_agg_binary_models/estroberta_SVC_main_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf12 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_agg_binary_models/estroberta_SVC_mean_concat.pkl', 'rb')) # TEHTUD, TEHTUD

#run_pipeline(clf1, './masters_thesis/durations/predict_rkogu', 'EstBERT', 'RF', 'main_last', 'all', X_test_main_last, y_test_last)
#run_pipeline(clf2, './masters_thesis/durations/predict_rkogu', 'EstBERT', 'RF', 'mean_last', 'all', X_test_mean_last, y_test_last)
run_pipeline(clf3, './masters_thesis/durations/predict_rkogu', 'EstRoBERTa', 'RF', 'main_last', 'all', X_test_main_last, y_test_last)
run_pipeline(clf4, './masters_thesis/durations/predict_rkogu', 'EstRoBERTa', 'RF', 'mean_last', 'all', X_test_mean_last, y_test_last)
#run_pipeline(clf5, './masters_thesis/durations/predict_rkogu', 'EstBERT', 'XGB', 'main_concat', 'agg_3', X_test_main_concat_tri, [labels2idx_3[label] for label in y_test_concat_tri])
#run_pipeline(clf6, './masters_thesis/durations/predict_rkogu', 'EstBERT', 'XGB', 'mean_concat', 'agg_3', X_test_mean_concat_tri, [labels2idx_3[label] for label in y_test_concat_tri])
run_pipeline(clf7, './masters_thesis/durations/predict_rkogu', 'EstRoBERTa', 'SVC', 'main_penult', 'agg_3', X_test_main_penult_tri, y_test_penult_tri)
run_pipeline(clf8, './masters_thesis/durations/predict_rkogu', 'EstRoBERTa', 'SVC', 'mean_penult', 'agg_3', X_test_mean_penult_tri, y_test_penult_tri)
#run_pipeline(clf9, './masters_thesis/durations/predict_rkogu', 'EstBERT', 'XGB', 'main_concat', 'agg_2', X_test_main_concat_bin, [labels2idx_2[label] for label in y_test_concat_bin])
#run_pipeline(clf10, './masters_thesis/durations/predict_rkogu', 'EstBERT', 'XGB', 'mean_concat', 'agg_2', X_test_mean_concat_bin, [labels2idx_2[label] for label in y_test_concat_bin])
run_pipeline(clf11, './masters_thesis/durations/predict_rkogu', 'EstRoBERTa', 'SVC', 'main_concat', 'agg_2', X_test_main_concat_bin, y_test_concat_bin)
run_pipeline(clf12, './masters_thesis/durations/predict_rkogu', 'EstRoBERTa', 'SVC', 'mean_concat', 'agg_2', X_test_mean_concat_bin, y_test_concat_bin)
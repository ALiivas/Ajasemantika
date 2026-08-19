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
    fn_path = "model_data/TimeML/"
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


def get_event_main_embeddings(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmuste-ajaväljendite peasõnade embeddingud ning tlink labelid.
    """
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
            tlink_labels.append(tlink["rel_type"])
                
    assert len(tlink_labels) == len(embed), "different list lengths"
    return embed, tlink_labels

def get_event_mean_embeddings(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmusfraaside sõnade embeddingute aritmeetilised keskmised väärtused ning kestuste labelid.
    """
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
            tlink_labels.append(tlink["rel_type"])                   
    assert len(tlink_labels) == len(embed), "different list lengths"
    return embed, tlink_labels
 
        
def get_event_avg_embeddings(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmusfraaside sõnade embeddingute kaalutud keskmised väärtused ning kestuste labelid.
    """
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
            embed.append(np.average(event_phrase_embed, 0))
            tlink_labels.append(tlink["rel_type"])           
    assert len(tlink_labels) == len(embed), "different list lengths"   
    return embed, tlink_labels

def get_event_mean_embeddings_agg(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmusfraaside sõnade embeddingute aritmeetilised keskmised väärtused ning kestuste labelid.
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
            tlink_labels.append(agg_dct[tlink["rel_type"]])                   
    assert len(tlink_labels) == len(embed), "different list lengths"
    return embed, tlink_labels
 
        
def get_event_avg_embeddings_agg(text_list, text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmusfraaside sõnade embeddingute kaalutud keskmised väärtused ning kestuste labelid.
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
            embed.append(np.average(event_phrase_embed, 0))
            tlink_labels.append(agg_dct[tlink["rel_type"]])           
    assert len(tlink_labels) == len(embed), "different list lengths"   
    return embed, tlink_labels


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
    with open(f'{save_path}/horisont_accuracy.txt', 'a') as f:
        f.write(f'{model_type}, {clf_name}, {embed_type}, {agg_type} accuracy: {acc_score}\n')
    
# tekstid ja embeddingud        
test_texts, test_concat, test_add = read_articles('rkogu/', 'masters_thesis/tlinks_ev_dct/embeddings/RoBERTa_embed_temp_facts/')

# viimased neli kihti liidetuna (konkatenatsioon)
X_test_mean_concat, y_test_mean_concat = get_event_mean_embeddings(test_texts, test_concat, None)
X_test_avg_concat, y_test_avg_concat = get_event_avg_embeddings(test_texts, test_concat, None)
X_test_mean_concat_tri, y_test_mean_concat_tri = get_event_mean_embeddings_tri(test_texts, test_concat, None)
X_test_avg_concat_tri, y_test_avg_concat_tri = get_event_avg_embeddings_tri(test_texts, test_concat, None)
X_test_mean_concat_bin, y_test_mean_concat_bin = get_event_mean_embeddings_bin(test_texts, test_concat, None)
X_test_avg_concat_bin, y_test_avg_concat_bin = get_event_avg_embeddings_bin(test_texts, test_concat, None)

# eelviimane kiht
X_test_mean_penult, y_test_mean_penult = get_event_mean_embeddings(test_texts, test_concat, 'penultimate')
X_test_avg_penult, y_test_avg_penult = get_event_avg_embeddings(test_texts, test_concat, 'penultimate')
X_test_mean_penult_tri, y_test_mean_penult_tri = get_event_mean_embeddings_tri(test_texts, test_concat, 'penultimate')
X_test_avg_penult_tri, y_test_avg_penult_tri = get_event_avg_embeddings_tri(test_texts, test_concat, 'penultimate')
X_test_mean_penult_bin, y_test_mean_penult_bin = get_event_mean_embeddings_bin(test_texts, test_concat, 'penultimate')
X_test_avg_penult_bin, y_test_avg_penult_bin = get_event_avg_embeddings_bin(test_texts, test_concat, 'penultimate')

# viimane kiht
X_test_mean_last, y_test_mean_last = get_event_mean_embeddings(test_texts, test_concat, 'last')
X_test_avg_last, y_test_avg_last = get_event_avg_embeddings(test_texts, test_concat, 'last')
X_test_mean_last_tri, y_test_mean_last_tri = get_event_mean_embeddings_tri(test_texts, test_concat, 'last')
X_test_avg_last_tri, y_test_avg_last_tri = get_event_avg_embeddings_tri(test_texts, test_concat, 'last')
X_test_mean_last_bin, y_test_mean_last_bin = get_event_mean_embeddings_bin(test_texts, test_concat, 'last')
X_test_avg_last_bin, y_test_avg_last_bin = get_event_avg_embeddings_bin(test_texts, test_concat, 'last')


clf1 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_models/estbert_RF_mean_penult.pkl', 'rb')) # TEHTUD, TEHTUD
clf2 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_models/estbert_RF_avg_penult.pkl', 'rb')) # TEHTUD, TEHTUD
clf3 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_models/estroberta_RF_mean_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf4 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_models/estroberta_RF_avg_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf5 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_agg_models/estbert_XGB_mean_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf6 = pickle.load(open('./masters_thesis/durations/estbert_ev_durations_agg_models/estbert_XGB_avg_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf7 = pickle.load(open('./masters_thesis/durations/estroberta_ev_durations_agg_models/estroberta_MLP_mean_concat.pkl', 'rb')) # TEHTUD, TEHTUD

#run_pipeline(clf1, './masters_thesis/durations/predict_horisont', 'EstBERT', 'RF', 'mean_penult', 'all', X_test_mean_penult, y_test_mean_penult)
#run_pipeline(clf2, './masters_thesis/durations/predict_horisont', 'EstBERT', 'RF', 'avg_penult', 'all', X_test_avg_penult, y_test_avg_penult)
#run_pipeline(clf3, './masters_thesis/durations/predict_horisont', 'EstRoBERTa', 'RF', 'mean_concat', 'all', X_test_mean_concat, y_test_mean_concat)
#run_pipeline(clf4, './masters_thesis/durations/predict_horisont', 'EstRoBERTa', 'RF', 'avg_concat', 'all', X_test_avg_concat, y_test_avg_concat)
#run_pipeline(clf5, './masters_thesis/durations/predict_horisont', 'EstBERT', 'XGB', 'mean_last', 'agg_3', X_test_mean_last_tri, [labels2idx_3[label] for label in y_test_mean_last_tri])
#run_pipeline(clf6, './masters_thesis/durations/predict_horisont', 'EstBERT', 'XGB', 'avg_last', 'agg_3', X_test_avg_last_tri, [labels2idx_3[label] for label in y_test_avg_last_tri])
#run_pipeline(clf7, './masters_thesis/durations/predict_horisont', 'EstRoBERTa', 'MLP', 'mean_concat', 'agg_3', X_test_mean_concat_tri, y_test_mean_concat_tri)
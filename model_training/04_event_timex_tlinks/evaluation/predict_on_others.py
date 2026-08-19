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
    fn_path = "./masters_thesis/tlinks_ev_tim/model_data/TempFact/"
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

# finds timex phrases of given sentence, if they exist in tlink layer
def get_sentence_phrases(text_obj, sentence, tlink_layer_name):
    phrase_matches = []
    for i in range(len(text_obj[tlink_layer_name])):
        phrase = []
        cur_tlink = text_obj[tlink_layer_name][i]
        for j in range(len(cur_tlink.b_text.base_span)):
            for word in sentence.words:
                if word.base_span == cur_tlink.b_text.base_span[j]:
                    phrase.append(word)
        if len(phrase) > 0:
            phrase_matches.append(phrase)
    return phrase_matches
    
# gets timex phrases' main word
def get_sentence_phrase_main_word(phrase, text_obj, current_word):
    current_word_span = text_obj.words.get(current_word)
    if current_word_span in phrase:
        return current_word_span
    else:
        for child in current_word.children:
            #print(text_obj.words.get(current_word))
            #print(f"current word: {current_word}, child: {current_word.children[i]}")
            current_word_span = get_sentence_phrase_main_word(phrase, text_obj, child)
            if current_word_span in phrase:
                return current_word_span

def get_event_timex_embeddings(tempfact_text_list, tempfact_text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmuste-ajaväljendite peasõnade embeddingud ning tlink labelid.
    """            
    event_main_embed = []
    timex_main_embed = []
    event_timex_mean_embed = []
    tlinks = []
    peasonu_kokku = 0
    #lausefraase_kokku = 0
    
    # ajafaktide korpus
    for text_idx, text in enumerate(tempfact_text_list):
        # leiame juba ette ära kõik teksti ajaväljendifraaside peasõnad
        all_timexes_main_words = []
        for sentence in text.sentences:
            sent_phrases = get_sentence_phrases(text, sentence, "tlinks")
            #lausefraase_kokku+=len(sent_phrases)
            sentence_root = None
            for j in range(len(sentence.stanza_syntax)):
                if sentence.stanza_syntax.deprel[j] == 'root':
                    sentence_root = sentence.stanza_syntax[j]        
            sent_timexes_main_words = []
            for phrase in sent_phrases:
                if len(phrase) == 1:
                    sent_timexes_main_words.append(phrase[0])
                else:
                    sent_timexes_main_words.append(get_sentence_phrase_main_word(phrase, text, sentence_root))
            assert len(sent_timexes_main_words) == len(sent_phrases), "Different number of timexes and main words"
            all_timexes_main_words += sent_timexes_main_words
        peasonu_kokku+=len(all_timexes_main_words)
            
        # sündmusfraasi ja ajaväljendifraasi peasõna vektorid
        for idx1, tlink in enumerate(text["tlinks"]):
            ev_word_spans = [text.words.get(span) for span in tlink.a_text.base_span]
            tm_word_spans = [text.words.get(span) for span in tlink.b_text.base_span]
            # jätame entiteedid välja
            for span in ev_word_spans:
                for ent in text.entities:
                    if span in ent:
                        ev_word_spans = [None]
            # Üksikutel juhtudel on timexis sõnestusprobleem, jätame need välja
            if None in tm_word_spans or None in ev_word_spans:
                continue
            #sündusfraasi peasõna vektori saamiseks kasutame ära varasemalt valmistehtud peasõnade IOB-kihti
            ev_found = False
            for idx2, word in enumerate(text.gold_word_events_main):
                if word.nertag == 'B-EVENT' or word.nertag == 'I-EVENT':
                    if text.words.get(word[0]) in ev_word_spans:
                        ev_found = True
                        if layer_type:
                            # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                            if layer_type == "penultimate":
                                event_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                            # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                            elif layer_type == "last":
                                event_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                        else:
                            event_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding))
                        break
            
            if not ev_found:
                continue
            
            # võtame timexi peasõna vektori
            tm_found = False           
            for idx2, word in enumerate(text.words):
                if word in tm_word_spans and word in all_timexes_main_words:
                    tm_found = True
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            timex_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            timex_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                    else:
                        timex_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding))
                    break
            
            if not tm_found:
                continue
            
            # leiame sündmusfraasi ja ajaväljendifraasi sõnade vektorid
            event_phrase_embed = []
            timex_phrase_embed = []
            for idx2, word in enumerate(text.words):
                if word in ev_word_spans:
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            event_phrase_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            event_phrase_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                    else:
                        event_phrase_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding))
                elif word in tm_word_spans:
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            timex_phrase_embed.append(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768])
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            timex_phrase_embed.append(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:])
                    else:
                        timex_phrase_embed.append(tempfact_text_embed_list[text_idx][idx2].bert_embedding)
            
            # lisame sündmuse-ajaväljendi embeddingute aritmeetilise keskmise            
            event_timex_mean_embed.append(np.concatenate((np.mean(event_phrase_embed, 0), np.mean(timex_phrase_embed, 0))))
            # lisame tlink labeli
            tlinks.append(tlink["rel_type"][0])
            
    print(len(event_main_embed), len(timex_main_embed))
    print(f"timexite peasonu leitud kokku {peasonu_kokku}")
    #print(f"lausefraase leitud kokku {lausefraase_kokku}")
    event_timex_main_embed = [np.concatenate((event_main_embed[i], timex_main_embed[i])) for i in range(len(event_main_embed))]
    print(len(tlinks), len(event_timex_main_embed), len(event_timex_mean_embed))
    print(tlinks)
    assert len(tlinks) == len(event_timex_main_embed) == len(event_timex_mean_embed), "different list lengths"
    return event_timex_main_embed, event_timex_mean_embed, tlinks

def get_event_timex_embeddings_agg(tempfact_text_list, tempfact_text_embed_list, layer_type):
    """
    Leiab ja tagastab sündmuste-ajaväljendite peasõnade embeddingud ning tlink labelid.
    """ 
    agg_dct = {'SIMULTANEOUS': 'OVERLAP',
              'INCLUDES': 'OVERLAP',
              'IS_INCLUDED': 'OVERLAP',
              'BEFORE': 'BEFORE',
              'AFTER': 'AFTER',
              'VAGUE': 'VAGUE'}           
    event_main_embed = []
    timex_main_embed = []
    event_timex_mean_embed = []
    tlinks = []
    peasonu_kokku = 0
    #lausefraase_kokku = 0
    
    # ajafaktide korpus
    for text_idx, text in enumerate(tempfact_text_list):
        # leiame juba ette ära kõik teksti ajaväljendifraaside peasõnad
        all_timexes_main_words = []
        for sentence in text.sentences:
            sent_phrases = get_sentence_phrases(text, sentence, "tlinks")
            #lausefraase_kokku+=len(sent_phrases)
            sentence_root = None
            for j in range(len(sentence.stanza_syntax)):
                if sentence.stanza_syntax.deprel[j] == 'root':
                    sentence_root = sentence.stanza_syntax[j]        
            sent_timexes_main_words = []
            for phrase in sent_phrases:
                if len(phrase) == 1:
                    sent_timexes_main_words.append(phrase[0])
                else:
                    sent_timexes_main_words.append(get_sentence_phrase_main_word(phrase, text, sentence_root))
            assert len(sent_timexes_main_words) == len(sent_phrases), "Different number of timexes and main words"
            all_timexes_main_words += sent_timexes_main_words
        peasonu_kokku+=len(all_timexes_main_words)
            
        # sündmusfraasi ja ajaväljendifraasi peasõna vektorid
        for idx1, tlink in enumerate(text["tlinks"]):
            ev_word_spans = [text.words.get(span) for span in tlink.a_text.base_span]
            tm_word_spans = [text.words.get(span) for span in tlink.b_text.base_span]
            # jätame entiteedid välja
            for span in ev_word_spans:
                for ent in text.entities:
                    if span in ent:
                        ev_word_spans = [None]
            # Üksikutel juhtudel on timexis sõnestusprobleem, jätame need välja
            if None in tm_word_spans or None in ev_word_spans:
                continue
            #sündusfraasi peasõna vektori saamiseks kasutame ära varasemalt valmistehtud peasõnade IOB-kihti
            ev_found = False
            for idx2, word in enumerate(text.gold_word_events_main):
                if word.nertag == 'B-EVENT' or word.nertag == 'I-EVENT':
                    if text.words.get(word[0]) in ev_word_spans:
                        ev_found = True
                        if layer_type:
                            # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                            if layer_type == "penultimate":
                                event_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                            # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                            elif layer_type == "last":
                                event_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                        else:
                            event_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding))
                        break
            
            if not ev_found:
                continue
            
            # võtame timexi peasõna vektori
            tm_found = False           
            for idx2, word in enumerate(text.words):
                if word in tm_word_spans and word in all_timexes_main_words:
                    tm_found = True
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            timex_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            timex_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                    else:
                        timex_main_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding))
                    break
            
            if not tm_found:
                continue
            
            # leiame sündmusfraasi ja ajaväljendifraasi sõnade vektorid
            event_phrase_embed = []
            timex_phrase_embed = []
            for idx2, word in enumerate(text.words):
                if word in ev_word_spans:
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            event_phrase_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            event_phrase_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                    else:
                        event_phrase_embed.append(np.asarray(tempfact_text_embed_list[text_idx][idx2].bert_embedding))
                elif word in tm_word_spans:
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            timex_phrase_embed.append(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768])
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            timex_phrase_embed.append(tempfact_text_embed_list[text_idx][idx2].bert_embedding[-768:])
                    else:
                        timex_phrase_embed.append(tempfact_text_embed_list[text_idx][idx2].bert_embedding)
            
            # lisame sündmuse-ajaväljendi embeddingute aritmeetilise keskmise            
            event_timex_mean_embed.append(np.concatenate((np.mean(event_phrase_embed, 0), np.mean(timex_phrase_embed, 0))))
            # lisame tlink labeli
            tlinks.append(agg_dct[tlink["rel_type"][0]])
            
    print(len(event_main_embed), len(timex_main_embed))
    print(f"timexite peasonu leitud kokku {peasonu_kokku}")
    #print(f"lausefraase leitud kokku {lausefraase_kokku}")
    event_timex_main_embed = [np.concatenate((event_main_embed[i], timex_main_embed[i])) for i in range(len(event_main_embed))]
    print(len(tlinks), len(event_timex_main_embed), len(event_timex_mean_embed))
    print(tlinks)
    assert len(tlinks) == len(event_timex_main_embed) == len(event_timex_mean_embed), "different list lengths"
    return event_timex_main_embed, event_timex_mean_embed, tlinks


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
test_texts, test_concat, test_add = read_articles('horisont/', 'masters_thesis/tlinks_ev_tim/embeddings/RoBERTa_embed_temp_facts/')

# viimased neli kihti liidetuna (konkatenatsioon)
X_test_main_concat, X_test_mean_concat, y_test_concat = get_event_timex_embeddings(test_texts, test_concat, None)
#X_test_avg_concat, y_test_avg_concat = get_event_timex_avg_embeddings(test_texts, test_concat, None)
X_test_main_concat_agg, X_test_mean_concat_agg, y_test_concat_agg = get_event_timex_embeddings_agg(test_texts, test_concat, None)
#X_test_avg_concat_agg, y_test_avg_concat_agg = get_event_timex_avg_embeddings_agg(test_texts, test_concat, None)

# viimase nelja kihi summa
X_test_main_add, X_test_mean_add, y_test_add = get_event_timex_embeddings(test_texts, test_add, None)
#X_test_avg_add, y_test_avg_add = get_event_timex_avg_embeddings(test_texts, test_add, None)
X_test_main_add_agg, X_test_mean_add_agg, y_test_add_agg = get_event_timex_embeddings_agg(test_texts, test_add, None)
#X_test_avg_add_agg, y_test_avg_add_agg = get_event_timex_avg_embeddings_agg(test_texts, test_add, None)

# eelviimane kiht
X_test_main_penult, X_test_mean_penult, y_test_penult = get_event_timex_embeddings(test_texts, test_concat, 'penultimate')
#X_test_avg_penult, y_test_avg_penult = get_event_timex_avg_embeddings(test_texts, test_concat, 'penultimate')
X_test_main_penult_agg, X_test_mean_penult_agg, y_test_penult_agg = get_event_timex_embeddings_agg(test_texts, test_concat, 'penultimate')
#X_test_avg_penult_agg, y_test_avg_penult_agg = get_event_timex_avg_embeddings_agg(test_texts, test_concat, 'penultimate')

# viimane kiht
X_test_main_last, X_test_mean_last, y_test_last = get_event_timex_embeddings(test_texts, test_concat, 'last')
X_test_main_last_agg, X_test_mean_last_agg, y_test_last_agg = get_event_timex_embeddings_agg(test_texts, test_concat, 'last')

labels2idx = {'OVERLAP': 0,
              'BEFORE': 1,
              'AFTER': 2,
              'VAGUE': 3}


clf1 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estbert_ev_timex_models/estbert_MLP_main_last.pkl', 'rb')) # TEHTUD, TEHTUD
clf2 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estroberta_ev_timex_models/estroberta_MLP_main_add.pkl', 'rb')) # TEHTUD, TEHTUD
clf3 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estroberta_ev_timex_models/estroberta_MLP_main_penult.pkl', 'rb')) # TEHTUD, TEHTUD
clf4 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estbert_ev_timex_agg_models/estbert_SVC_mean_add.pkl', 'rb')) # TEHTUD, TEHTUD
clf5 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estroberta_ev_timex_agg_models/estroberta_MLP_main_concat.pkl', 'rb')) # TEHTUD, TEHTUD
clf6 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estroberta_ev_timex_agg_models/estroberta_SVC_main_add.pkl', 'rb')) # TEHTUD, TEHTUD
clf7 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estroberta_ev_timex_agg_models/estroberta_MLP_mean_penult.pkl', 'rb')) # TEHTUD, TEHTUD
clf8 = pickle.load(open('./masters_thesis/tlinks_ev_tim/estroberta_ev_timex_agg_models/estroberta_XGB_mean_penult.pkl', 'rb')) # TEHTUD, TEHTUD

#run_pipeline(clf1, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstBERT', 'MLP', 'main_last', 'all', X_test_main_last, y_test_last)
run_pipeline(clf2, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstRoBERTa', 'MLP', 'main_add', 'all', X_test_main_add, y_test_add)
run_pipeline(clf3, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstRoBERTa', 'MLP', 'main_penult', 'all', X_test_main_penult, y_test_penult)
#run_pipeline(clf4, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstBERT', 'SVC', 'mean_add', 'agg', X_test_mean_add_agg, y_test_add_agg)
run_pipeline(clf5, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstRoBERTa', 'MLP', 'main_concat', 'agg', X_test_main_concat_agg, y_test_concat_agg)
run_pipeline(clf6, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstRoBERTa', 'SVC  ', 'main_add', 'agg', X_test_main_add_agg, y_test_add_agg)
run_pipeline(clf7, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstRoBERTa', 'MLP', 'mean_penult', 'agg', X_test_mean_penult_agg, y_test_penult_agg)
run_pipeline(clf8, './masters_thesis/tlinks_ev_tim/predict_horisont', 'EstRoBERTa', 'XGB', 'mean_penult', 'agg', X_test_mean_penult_agg, [labels2idx[label] for label in y_test_penult_agg])
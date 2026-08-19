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
    fn_path1 = "model_data/TimeML/"
    fn_path2 = "model_data/TempFact/"

    timeml_train_files = []
    timeml_test_files = []
    tempfact_train_files = []
    tempfact_test_files = []

    for filename in os.listdir(fn_path1+"train/"):
        timeml_train_files.append(json_to_text(file=os.path.join(fn_path1+"train/", filename)))
    for filename in os.listdir(fn_path1+"dev/"): # lisame dev failid train failide juurde
        timeml_train_files.append(json_to_text(file=os.path.join(fn_path1+"dev/", filename)))
    for filename in os.listdir(fn_path1+"test/"):
        timeml_test_files.append(json_to_text(file=os.path.join(fn_path1+"test/", filename)))
    for filename in os.listdir(fn_path2+"train_larger/"): # lisame dev failid train failide juurde
        tempfact_train_files.append(json_to_text(file=os.path.join(fn_path2+"train_larger/", filename)))
    for filename in os.listdir(fn_path2+"test/"):
        tempfact_test_files.append(json_to_text(file=os.path.join(fn_path2+"test/", filename)))
    
    # embeddingutega lugemine train-dev-test hulkadesse    
    embed_path_estbert1 = "embeddings/BERT_embed_EstTimeML/"
    embed_path_estbert2 = "embeddings/BERT_embed_temp_facts/"
    
    timeml_train_files_concat = []
    timeml_test_files_concat = []
    timeml_train_files_add = []
    timeml_test_files_add = []
    
    tempfact_train_files_concat = []
    tempfact_test_files_concat = []
    tempfact_train_files_add = []
    tempfact_test_files_add = []
    
    for text in timeml_train_files:
        filename = text.meta['filename']
        timeml_train_files_concat.append(json_to_layer(text, file=f"{embed_path_estbert1}{filename}_embed_concat.json"))
        timeml_train_files_add.append(json_to_layer(text, file=f"{embed_path_estbert1}{filename}_embed_add.json"))
    for text in timeml_test_files:
        filename = text.meta['filename']
        timeml_test_files_concat.append(json_to_layer(text, file=f"{embed_path_estbert1}{filename}_embed_concat.json"))
        timeml_test_files_add.append(json_to_layer(text, file=f"{embed_path_estbert1}{filename}_embed_add.json"))
        
    for text in tempfact_train_files:
        filename = text.meta['filename']
        tempfact_train_files_concat.append(json_to_layer(text, file=f"{embed_path_estbert2}{filename}_embed_concat.json"))
        tempfact_train_files_add.append(json_to_layer(text, file=f"{embed_path_estbert2}{filename}_embed_add.json"))
    for text in tempfact_test_files:
        filename = text.meta['filename']
        tempfact_test_files_concat.append(json_to_layer(text, file=f"{embed_path_estbert2}{filename}_embed_concat.json"))
        tempfact_test_files_add.append(json_to_layer(text, file=f"{embed_path_estbert2}{filename}_embed_add.json"))
    
    #print("Lugesin sisse", len(train_files_concat), len(train_files_add), "treeningandmete faili.")
    #print("Lugesin sisse", len(test_files_concat), len(test_files_add), "testandmete faili.")
    
    return timeml_train_files, timeml_test_files, tempfact_train_files, tempfact_test_files, timeml_train_files_concat, timeml_train_files_add, timeml_test_files_concat, timeml_test_files_add, tempfact_train_files_concat, tempfact_train_files_add, tempfact_test_files_concat, tempfact_test_files_add

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

def get_event_timex_embeddings(timeml_text_list, timeml_text_embed_list, tempfact_text_list, tempfact_text_embed_list, layer_type):
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
    
    # timeml korpus
    for text_idx, text in enumerate(timeml_text_list):
        # leiame juba ette ära kõik teksti ajaväljendifraaside peasõnad
        all_timexes_main_words = []
        for sentence in text.sentences:
            sent_phrases = get_sentence_phrases(text, sentence, "event_timex_tlinks")
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
            
        # sündmusfraasi ja ajaväljendifraasi peasõna vektorid; vektorite aritmeetiline keskmine
        for idx1, tlink in enumerate(text["event_timex_tlinks"]):
            ev_word_spans = [text.words.get(span) for span in tlink.a_text.base_span]
            tm_word_spans = [text.words.get(span) for span in tlink.b_text.base_span]
            # Üksikutel juhtudel on timexis sõnestusprobleem, jätame need välja
            if None in tm_word_spans:
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
                                event_main_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                            # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                            elif layer_type == "last":
                                event_main_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                        else:
                            event_main_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding))
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
                            timex_main_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            timex_main_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                    else:
                        timex_main_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding))
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
                            event_phrase_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768]))
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            event_phrase_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding[-768:]))
                    else:
                        event_phrase_embed.append(np.asarray(timeml_text_embed_list[text_idx][idx2].bert_embedding))
                elif word in tm_word_spans:
                    if layer_type:
                        # kui eelviimane kiht, siis võtame vektoriks bert_embedding[-1536:-768]
                        if layer_type == "penultimate":
                            timex_phrase_embed.append(timeml_text_embed_list[text_idx][idx2].bert_embedding[-1536:-768])
                        # kui viimane kiht, siis võtame vektoriteks bert_embedding[-768:]
                        elif layer_type == "last":
                            timex_phrase_embed.append(timeml_text_embed_list[text_idx][idx2].bert_embedding[-768:])
                    else:
                        timex_phrase_embed.append(timeml_text_embed_list[text_idx][idx2].bert_embedding)
            
            # lisame sündmuse-ajaväljendi embeddingute aritmeetilise keskmise            
            event_timex_mean_embed.append(np.concatenate((np.mean(event_phrase_embed, 0), np.mean(timex_phrase_embed, 0))))
            # lisame tlink labeli
            tlinks.append(agg_dct[tlink["rel_type"]])
    
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


def run_pipeline(clf, save_path, clf_name, embed_type, X_train, y_train, X_test, y_test):
    """
    Treenib ja leiab mudeli õigsuse testandmestikul.
    """
    # treenime mudeli
    clf.fit(X_train, y_train)
    if clf_name != "dummy":
        # salvestame mudeli faili
        pickle.dump(clf, open(f'estbert_ev_timex_agg_models/estbert_{clf_name}_{embed_type}.pkl', 'wb'))
    # ennustame testandmestiku labelid
    y_pred = clf.predict(X_test)
    # leiame õigsuse
    acc_score = accuracy_score(y_test, y_pred)
    clsf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    # salvestame tulemused
    clsf_report.to_csv(f'{save_path}/{clf_name}_{embed_type}_clsf_report.csv', index= True)
    with open(f'{save_path}/estbert_ev_timex_agg_accuracy.txt', 'a') as f:
        f.write(f'{clf_name}, {embed_type} accuracy: {acc_score}\n')
    
# tekstid ja embeddingud        
timeml_train_texts, timeml_test_texts, tempfact_train_texts, tempfact_test_texts, timeml_train_concat, timeml_train_add, timeml_test_concat, timeml_test_add, tempfact_train_concat, tempfact_train_add, tempfact_test_concat, tempfact_test_add = read_articles()

# viimase nelja kihi konkatenatsioon
X_train_main_concat, X_train_mean_concat, y_train_concat = get_event_timex_embeddings(timeml_train_texts, timeml_train_concat, tempfact_train_texts, tempfact_train_concat, None)
X_test_main_concat, X_test_mean_concat, y_test_concat = get_event_timex_embeddings(timeml_test_texts, timeml_test_concat, tempfact_test_texts, tempfact_test_concat, None)

# viimase nelja kihi summa
X_train_main_add, X_train_mean_add, y_train_add = get_event_timex_embeddings(timeml_train_texts, timeml_train_add, tempfact_train_texts, tempfact_train_add, None)
X_test_main_add, X_test_mean_add, y_test_add = get_event_timex_embeddings(timeml_test_texts, timeml_test_add, tempfact_test_texts, tempfact_test_add, None)

# eelviimane kiht
X_train_main_penult, X_train_mean_penult, y_train_penult = get_event_timex_embeddings(timeml_train_texts, timeml_train_concat, tempfact_train_texts, tempfact_train_concat, 'penultimate')
X_test_main_penult, X_test_mean_penult, y_test_penult = get_event_timex_embeddings(timeml_test_texts, timeml_test_concat, tempfact_test_texts, tempfact_test_concat, 'penultimate')

# viimane kiht
X_train_main_last, X_train_mean_last, y_train_last = get_event_timex_embeddings(timeml_train_texts, timeml_train_concat, tempfact_train_texts, tempfact_train_concat, 'last')
X_test_main_last, X_test_mean_last, y_test_last = get_event_timex_embeddings(timeml_test_texts, timeml_test_concat, tempfact_test_texts, tempfact_test_concat, 'last')

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

labels2idx = {'OVERLAP': 0,
              'BEFORE': 1,
              'AFTER': 2,
              'VAGUE': 3}

# ---------treenimine ja tulemused-----------

# nelja viimase kihi konkatenatsioon
# peasõna vektorid
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'main_concat', X_train_main_concat, [labels2idx[label] for label in y_train_concat], X_test_main_concat, [labels2idx[label] for label in y_test_concat])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'main_concat', X_train_main_concat, y_train_concat, X_test_main_concat, y_test_concat)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'mean_concat', X_train_mean_concat, [labels2idx[label] for label in y_train_concat], X_test_mean_concat, [labels2idx[label] for label in y_test_concat])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'mean_concat', X_train_mean_concat, y_train_concat, X_test_mean_concat, y_test_concat)

# nelja viimase kihi summa
# peasõna vektorid
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'main_add', X_train_main_add, [labels2idx[label] for label in y_train_add], X_test_main_add, [labels2idx[label] for label in y_test_add])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'main_add', X_train_main_add, y_train_add, X_test_main_add, y_test_add)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'mean_add', X_train_mean_add, [labels2idx[label] for label in y_train_add], X_test_mean_add, [labels2idx[label] for label in y_test_add])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'mean_add', X_train_mean_add, y_train_add, X_test_mean_add, y_test_add)

# eelviimane kiht
# peasõna vektorid
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'main_penult', X_train_main_penult, [labels2idx[label] for label in y_train_penult], X_test_main_penult, [labels2idx[label] for label in y_test_penult])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'main_penult', X_train_main_penult, y_train_penult, X_test_main_penult, y_test_penult)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'mean_penult', X_train_mean_penult, y_train_penult, X_test_mean_penult, y_test_penult)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'mean_penult', X_train_mean_penult, y_train_penult, X_test_mean_penult, y_test_penult)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'mean_penult', X_train_mean_penult, y_train_penult, X_test_mean_penult, y_test_penult)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'mean_penult', X_train_mean_penult, [labels2idx[label] for label in y_train_penult], X_test_mean_penult, [labels2idx[label] for label in y_test_penult])

# viimane kiht
# peasõna vektorid
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'main_last', X_train_main_last, [labels2idx[label] for label in y_train_last], X_test_main_last, [labels2idx[label] for label in y_test_last])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'main_last', X_train_main_last, y_train_last, X_test_main_last, y_test_last)

# fraasi vektorite aritmeetiline keskmine
run_pipeline(svc_clf, 'estbert_ev_timex_agg_res', 'SVC', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)
run_pipeline(rf_clf, 'estbert_ev_timex_agg_res', 'RF', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)
run_pipeline(mlp_clf, 'estbert_ev_timex_agg_res', 'MLP', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)
run_pipeline(xgb_clf, 'estbert_ev_timex_agg_res', 'XGB', 'mean_last', X_train_mean_last, [labels2idx[label] for label in y_train_last], X_test_mean_last, [labels2idx[label] for label in y_test_last])
run_pipeline(dummy_clf, 'estbert_ev_timex_agg_res', 'dummy', 'mean_last', X_train_mean_last, y_train_last, X_test_mean_last, y_test_last)

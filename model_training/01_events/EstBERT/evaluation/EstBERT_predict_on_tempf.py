import os
import json

import estnltk
from estnltk.converters import json_to_text, text_to_json
from estnltk_neural.taggers.embeddings.bert.bert_tokens_to_words_rewriter import BertTokens2WordsRewriter

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nervaluate import Evaluator

def save_predictions(tempf_path, corpus_texts, sentence_starts, tokens_w_spans, pred_tag_values):
    
    def create_bert_token_layer(text_obj, sentence_starts, tokens_w_spans, pred_labels, layer_name):
        event_pred_layer = estnltk.Layer(name=layer_name, text_object=text_obj, attributes=['token', 'nertag'], ambiguous=True)
        for i, token_list in enumerate(tokens_w_spans):
            for j, token in enumerate(token_list):
                if token[0] is None and token[1] is None:
                    # Skip special tokens (e.g. [CLS], [SEP])
                    continue
                try:
                    pred_label = pred_labels[i][j]
                except:
                    #print(pred_labels)
                    #print(tokens_w_spans)
                    print(len(pred_labels))
                    print(i)
                    print(j)
                    break
                try:    
                    token_start = sentence_starts[i]+token[0]
                except:
                    print(i)
                    print(sentence_starts)
                    print(len(tokens_w_spans))
                    print(tokens_w_spans)
                    print(sentence_starts[i])
                    print("Token: ", token)
                    
                token_end = sentence_starts[i]+token[1]
                attributes = {'token': token[2], 'nertag': pred_label}
                event_pred_layer.add_annotation((token_start, token_end), **attributes)  
        return event_pred_layer
    
    def rewriter_decorator(text_obj, sharing_words, shared_bert_tokens):
        return {'estbert_tokens': [t.text for t in shared_bert_tokens],
            'nertag': [n.nertag for n in shared_bert_tokens][0][0]}
    
    rewriter = BertTokens2WordsRewriter('estbert_tokens', 
                                    input_words_layer = 'words', 
                                    output_attributes = ('estbert_tokens', 'nertag'), 
                                    output_layer = 'estbert_tokens_of_words',
                                    enveloping = False,
                                    decorator = rewriter_decorator)
    
    for idx, text in enumerate(corpus_texts):
        fname = text.meta['filename']
        text.add_layer(create_bert_token_layer(text, sentence_starts[idx], tokens_w_spans[fname], pred_tag_values[fname], 'estbert_tokens'))
        print(fname)
        rewriter.tag(text)
        save_path = tempf_path.strip('/')+"_pred/"
        #save_path = tempf_path
        filename = save_path + fname + '_pred.json'
        text_to_json(text, file=filename)


def predict_on_tempf(corpus_path, dtype, model, tokenizer, layer, event_classes):
    tempf_path = corpus_path
    tempf_texts = []

    for filename in os.listdir(tempf_path):
        text_obj = json_to_text(file=os.path.join(tempf_path, filename))
        tempf_texts.append(text_obj)

    # -- Method for splitting corpus data into sentences with corresponding labels and IDs
    def split_data(text_objects, layer):
        filenames = []
        sentence_starts = []
        sentences = []
        tokens = []
        labels = []
    
        for text in text_objects:
            sent_starts = []
            for sentence in text.sentences:
                sent = []
                sent_labels = []
                for word in sentence.words:
                    for w in text[layer]:
                        if word == w[0]:
                            sent.append(w[0].text)
                            sent_labels.append(w.nertag)
                filenames.append(text.meta['filename'])
                sent_starts.append(sentence.start)
                sentences.append(sentence.enclosing_text)
                tokens.append(sent)
                labels.append(sent_labels)
                
            sentence_starts.append(sent_starts)
    
        assert len(sentences) == len(tokens) == len(labels), "Different number of sentences and corresponding labels"    
        return filenames, sentence_starts, sentences, tokens, labels

    filenames, sentence_starts, sentences, tokens, labels = split_data(tempf_texts, layer)
    
    label_list = list(set([l for label in labels for l in label]))
    label_list.append('PAD')

    # based on HuggingFace tutorial (https://huggingface.co/docs/transformers/tasks/token_classification)
    def tokenize_and_align_labels(texts, true_labels):
        tokenized_inputs = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, is_split_into_words=True)
 
        aligned_labels = []
    
        for i, text in enumerate(texts):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(true_labels[i][word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx       
            aligned_labels.append(label_ids)      
        return tokenized_inputs, aligned_labels
    
    # tokenize with tokenizer
    tokenizer = tokenizer
    tokenized_inputs, aligned_labels = tokenize_and_align_labels(tokens, labels)
    
    all_pred_tag_values = []
    all_pred_tokens_w_spans = []   
    all_preds = [] # kõik puhastatud ennustused
    cleaned_aligned_labels = [] # puhastatud labelid
    # model
    model = model
    with torch.no_grad():
        logits = model(**tokenized_inputs).logits
        predictions = torch.argmax(logits, dim=2)
        
        for i, pred in enumerate(predictions):
            #pred_tokens = tokenized_inputs.tokens(batch_index=i)
            # leiame iga tokeni algus- ja lõpuindeksi algses lauses
            tokens_w_spans = []
            include_spanless = True
            batch_encoding = tokenizer(sentences[i])
            for token_id, token in enumerate(batch_encoding.tokens()):
                char_span = batch_encoding.token_to_chars(token_id)
                if char_span is not None:
                    tokens_w_spans.append( (char_span.start, char_span.end, token) )
                elif include_spanless:
                    tokens_w_spans.append( (None, None, token) )

            all_pred_tokens_w_spans.append(tokens_w_spans)
            prediction_tag_values = [model.config.id2label[pred.item()] for pred in predictions[i]]
            all_pred_tag_values.append(prediction_tag_values)
            final_predictions = [prediction for (prediction, label) in zip(prediction_tag_values, aligned_labels[i]) if label != -100]
            final_labels = [label for label in aligned_labels[i] if label != -100]
            all_preds.append(final_predictions)
            cleaned_aligned_labels.append(final_labels)
            assert len(final_predictions) == len(final_labels), "Eri pikkusega tõelised ja ennustatud labelid"

    #viime kokku ennustused ja failinimed  
    file_tokens_w_spans = {}
    file_pred_tag_values = {}
    for filename in filenames:
        file_tokens_w_spans.update({filename: []})
        file_pred_tag_values.update({filename: []})
    
    for f, s, t in zip(filenames, all_pred_tokens_w_spans, all_pred_tag_values):
        file_tokens_w_spans[f].append(s)
        file_pred_tag_values[f].append(t)
    
    nervaluate = Evaluator(cleaned_aligned_labels, all_preds, tags=event_classes, loader='list')
    results, results_by_tag, result_indices, result_indices_by_tag = nervaluate.evaluate()
    
    # -- Saving evaluation results to JSON
    #with open(f'masters_thesis/estbert_events_param_tuning_pred_tempf_{dtype}.json', 'w') as f: 
    #    json.dump([results, results_by_tag, result_indices, result_indices_by_tag], f)
    
    # -- creating layer of predictions and saving
    save_predictions(tempf_path, tempf_texts, sentence_starts, file_tokens_w_spans, file_pred_tag_values)
        
    
tokenizer = AutoTokenizer.from_pretrained('./masters_thesis/estbert_events_param_tuning_best/best')
model = AutoModelForTokenClassification.from_pretrained('./masters_thesis/estbert_events_param_tuning_best/best')
event_classes = ['EVENT']

predict_on_tempf('masters_thesis/timeml_final_test/', 'news', model, tokenizer, 'gold_word_events', event_classes)
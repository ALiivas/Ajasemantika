import os
import json
from estnltk.converters import json_to_text

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nervaluate import Evaluator

def predict_on_tempf(model, tokenizer, layer):
    tempf_path = 'masters_thesis/temporal_facts_corpus_json_no_horisont/'

    tempf_texts = []

    for filename in os.listdir(tempf_path):
        text_obj = json_to_text(file=os.path.join(tempf_path, filename))
        tempf_texts.append(text_obj)

    # -- Method for splitting corpus data into sentences with corresponding labels and IDs
    def split_data(text_objects, layer):
        filenames = []
        sentences = []
        tokens = []
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
                filenames.append(text.meta['filename'])
                sentences.append(' '.join(sent))
                tokens.append(sent)
                labels.append(sent_labels)
    
        assert len(sentences) == len(tokens) == len(labels), "Different number of sentences and corresponding labels"
       
        return filenames, sentences, tokens, labels

    filenames, sentences, tokens, labels = split_data(tempf_texts, layer)

    # tokenizer
    tokenizer = tokenizer
    inputs_list = []
    for sent in sentences:
        inputs = tokenizer.encode_plus(
            sent,
            truncation=True,
            padding=True,
            max_length=75,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        #inputs = tokenizer(sent, return_tensors="pt")
        inputs_list.append(inputs)
    #inputs = [tokenizer(sent, return_tensors="pt") for sent in sentences]
    #print(inputs_list[0])
    
    all_preds = []
    all_pred_tag_values = []
    all_tokens = []
    # model
    model = model
    for input in inputs_list:
        with torch.no_grad():
            logits = model(**input).logits

        #print(input)
        tokens = [tokenizer.convert_ids_to_tokens(token) for token in input['input_ids']][0]
        #print(tokens)
        predictions = torch.argmax(logits, dim=2)
        prediction_tag_values = [model.config.id2label[t.item()] for t in predictions[0]]
        all_pred_tag_values.append(prediction_tag_values)
        cleaned_tokens = []
        cleaned_labels = []
        all_tokens.append(tokens)
        for i in range(len(tokens)):
            if not tokens[i].startswith("▁") and tokens[i] != '<s>' and tokens[i] != '</s>': #roberta puhul on token.startswith("_"), aga see on ainult esimese sõna alguses JA see sümbol tuleb eraldi kopeerida kusagilt (pole tavaline alakriips)
                # kui sõna esimes(t)ele juppidele on ennustatud vale label, otsime ülejäänud juppide seast, kas leidub õige label
                if cleaned_labels[-1] != labels[len(cleaned_labels)-1]:
                    cleaned_labels[-1] = prediction_tag_values[i]
                cleaned_tokens[-1] = cleaned_tokens[-1] + tokens[i]
            elif tokens[i] == '<s>' or tokens[i] == '</s>':
                continue
            else:
                cleaned_labels.append(prediction_tag_values[i])
                cleaned_tokens.append(tokens[i][1:])
        all_preds.append(cleaned_labels)
    #print(all_tokens[0])
    #print(all_preds[0])
    #print(labels[0])
    #print(sentences[0])
    
    for i in range(len(all_preds)):
        if all_preds[i] != labels[i]:
            with open("Est-RoBERTa_predict_on_tempf_values_w_agg_classes_no_modal_no_horisont.txt", "a", encoding="utf-8") as f:
                f.write(filenames[i]+"\n")
                for j in range(len(all_tokens[i])):
                    f.write(str(all_tokens[i][j])+"\t"+all_pred_tag_values[i][j]+"\n")
                f.write(str(labels[i])+"\n")
                f.write("\n")
    
    nervaluate = Evaluator(labels, all_preds, tags=['EVENT_ARG_CLASS', 'EVENT_OCCURRENCE', 'EVENT_STATE'], loader='list')
    results, results_by_tag, result_indices, result_indices_by_tag = nervaluate.evaluate()
    
    # -- Saving evaluation results to JSON
    with open('masters_thesis/est-roberta_events_w_agg_class_no_modal_param_tuning_pred_tempf_no_horisont.json', 'w') as f: 
        json.dump([results, results_by_tag, result_indices, result_indices_by_tag], f)  
    
tokenizer = AutoTokenizer.from_pretrained('./masters_thesis/est-roberta_events_w_agg_class_no_modal_param_tuning_best/best')
model = AutoModelForTokenClassification.from_pretrained('./masters_thesis/est-roberta_events_w_agg_class_no_modal_param_tuning_best/best')

predict_on_tempf(model, tokenizer, 'gold_word_events_w_agg_classes_no_modal')
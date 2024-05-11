# Script for converting article text to EstNLTK Text-object with layers
#
# Requirements: EstNLTK 1.7.2
# 
# -- imports
import estnltk
from estnltk import Text
from estnltk_core import RelationLayer

# -- method for creating event and timex layers
def create_event_and_timex_layers(text_obj, sentence_base_annotations, sentence_events, sentence_timex, mapping):
    event_classes = ['REPORTING', 'PERCEPTION', 'ASPECTUAL', 'I_ACTION', 'I_STATE', 'STATE', 'MODAL',
                     'OCCURRENCE', 'EVENT_CONTAINER', 'CAUSE']
    timex_types = ['DATE', 'TIME', 'DURATION', 'SET']
    
    event_layer = estnltk.Layer(name='gold_events',
                  text_object=text_obj,
                  attributes=['corpus_sentence_ID', 'corpus_word_ID','event_ID', 'expression', 'event_annotation',
                              'event_class'])
    timex_layer = estnltk.Layer(name='gold_timexes',
                  text_object=text_obj,
                  attributes=['corpus_sentence_ID', 'corpus_word_ID', 'timex_ID', 'expression', 'timex_annotation',
                              'type', 'value'], ambiguous=True)

    last_event_class = None
    last_timex_type = None
    last_timex_value = None
    previous_multiword = False
    estnltk_word_id = 0
    
    for sentence_id, sentence in enumerate(sentence_base_annotations):
        for word_id_in_sentence, word_info in enumerate(sentence):
            word = word_info[2]
            estnltk_word = text_obj.words[estnltk_word_id]
            startend = mapping.get((sentence_id, word_id_in_sentence))
            estnltk_startend = tuple([estnltk_word.start, estnltk_word.end])     
            event = sentence_events.get((str(sentence_id), str(word_id_in_sentence)))
            if sentence_timex:
                timex = sentence_timex.get((str(sentence_id), str(word_id_in_sentence)))
            else:
                timex = None
            
            if event:
                # if spans are not overlapping, start and end are synced with estNLTK words layer
                if estnltk_startend[0] <= startend[0] <= estnltk_startend[1] or startend[0] <= estnltk_startend[0] <= startend[1]:
                    startend = estnltk_startend        
                event_tag = event[0][0]
                e_class = event[0][2].split()[1]
                # if multiword
                if 'multiword="true"' in event[0][2].split():
                    # event class is in second position
                    if event[0][2].split()[1] not in event_classes:
                        # if multiword part with class is before multiword part without class in text, event class will be corrected
                        previous_class = None
                        for e in event_layer:
                            if e.event_ID==event_tag:
                                previous_class = e.event_annotation.split()[1]
                                break
                        if previous_class!=None and previous_class in event_classes:
                            e_class = previous_class
                        else:
                            previous_multiword = True
                    # if multiword part without class is before multiword part with class in text, event class will be corrected
                    if previous_multiword:
                        for e in event_layer:
                            if e.event_ID==event_tag and e.event_class not in event_classes and e_class in event_classes:
                                e.event_class=e_class
                                previous_multiword=False
                                
                event_layer.add_annotation((startend[0], startend[1]), corpus_sentence_ID=sentence_id,
                                           corpus_word_ID=word_id_in_sentence, event_ID=event_tag,
                                           expression=event[0][1], event_annotation=event[0][2],
                                           event_class=e_class)
            
            elif timex:
                # if spans are not overlapping, start and end are synced with estNLTK words layer
                if estnltk_startend[0] <= startend[0] <= estnltk_startend[1] or startend[0] <= estnltk_startend[0] <= startend[1]:
                    startend = estnltk_startend                
                token_start = startend[0]
                token_end = startend[1]
                for i in range(len(timex)):
                    timex_type = timex[i][2].split()[1]
                    # timex is probably part of multiword, if timex type does not exist in timex types
                    if timex_type in timex_types:
                        last_timex_type = timex_type
                        timex_value = timex[i][2].split()[2]
                        last_timex_value = timex_value
                    else:
                        timex_type = last_timex_type
                        timex_value = last_timex_value   
                        
                    timex_layer.add_annotation((token_start, token_end), corpus_sentence_ID=sentence_id,
                                               corpus_word_ID=word_id_in_sentence, timex_ID=timex[i][0],
                                               expression=timex[i][1], timex_annotation=timex[i][2],
                                               type=timex_type, value=timex_value)
                    
            if estnltk_word_id < len(text_obj.words)-1:
                estnltk_word_id+=1
    
    return event_layer, timex_layer


# -- method for creating gold_word_events layer with IOB-annotations
def create_gold_word_events_layer(text_obj):
    gold_word_events = estnltk.Layer(name="gold_word_events", text_object=text_obj, attributes=['nertag'],
                                     enveloping='words')
    last_event_tag = None
    multiword_event_tags = set()
    for word in text_obj.words:
        event = text_obj.gold_events.get(word)
        if event:
            is_multiword = 'multiword="true"' in event['event_annotation']
            event_tag = event['event_ID']
            if event_tag != last_event_tag and event_tag not in multiword_event_tags:
                gold_word_events.add_annotation([word.base_span], nertag='B-EVENT')
            else:
                gold_word_events.add_annotation([word.base_span], nertag='I-EVENT')
            if is_multiword:
                multiword_event_tags.add(event_tag)
            last_event_tag = event_tag
        else:
            gold_word_events.add_annotation([word.base_span], nertag='O')
    
    return gold_word_events


# -- method for creating gold_word_events layer with IOB-annotations and event classes
def create_gold_word_events_with_classes_layer(text_obj):
    gold_word_events_w_classes = estnltk.Layer(name="gold_word_events_w_classes", text_object=text_obj, attributes=['nertag'],
                                     enveloping='words')
    
    for i in range(len(text_obj.gold_word_events)):
        iob_word = text_obj.gold_word_events[i]
        event = None
        if iob_word['nertag'] == 'B-EVENT' or iob_word['nertag'] == 'I-EVENT':
            event = text_obj.gold_events.get(iob_word)
        if event:
            event_class = event['event_class']
            gold_word_events_w_classes.add_annotation([iob_word[0].base_span], nertag=iob_word['nertag']+'_'+event_class[0])
        else:
            gold_word_events_w_classes.add_annotation([iob_word[0].base_span], nertag='O')
    
    return gold_word_events_w_classes


# -- method for creating layer of gold event phrases
def create_gold_event_phrases(text_obj):
    gold_event_phrases = estnltk.Layer(name='gold_event_phrases',
                                      text_object=text_obj,
                                      attributes=['event_ID', 'expression', 'event_class'],
                                      enveloping='words', ambiguous=True)
    event_groups = text_obj.gold_events.groupby(['event_ID'], return_type='spans')
    for key, value in event_groups:
        phrase_spans = []
        for span in value:
            phrase_spans.append(span)
        gold_event_phrases.add_annotation([s.base_span for s in phrase_spans], event_ID=key[0], 
                                     expression=phrase_spans[0]['expression'], event_class=phrase_spans[0]['event_class'])
    
    return gold_event_phrases


# -- method for creating layer of gold timex phrases
def create_gold_timex_phrases(text_obj):
    gold_timex_phrases = estnltk.Layer(name='gold_timex_phrases',
                                      text_object=text_obj,
                                      attributes=['timex_ID', 'expression', 'type', 'value'],
                                      enveloping='words', ambiguous=True)
    timex_groups = text_obj.gold_timexes.groupby(['timex_ID'], return_type='spans')
    for key, value in timex_groups:
        phrase_spans = []
        for span in value:
            phrase_spans.append(span)
        gold_timex_phrases.add_annotation([s.base_span for s in phrase_spans], timex_ID=key[0], 
                                     expression=phrase_spans[0]['expression'][0], type=phrase_spans[0]['type'][0], 
                                          value=phrase_spans[0]['value'][0])
    
    return gold_timex_phrases


# -- method for creating event-timex TLINK relation layer
def create_event_timex_rel_layer(text_obj, relations):
    tlinks_layer = RelationLayer('event_timex_tlinks', span_names=['a_text', 'b_text'], 
                                 attributes=['a_ID', 'rel_type', 'b_ID', 'comment'], 
                                 display_order=['a_text', 'a_ID', 'rel_type', 'b_text', 'b_ID', 'comment'], 
                                 text_object=text_obj, enveloping='words')    
    if relations:
        last_ann = None
        for entity in relations:
            ann = relations.get(entity)[0]
            if last_ann==ann:
                last_ann == None
                continue      
            else:
                last_ann = ann
                a_ID = ann[0]
                rel_type = ann[1]
                b_ID = ann[2]
                comment = ann[3]           
                event_groups = text_obj.gold_events.groupby(['event_ID'], return_type='spans')
                timex_groups = text_obj.gold_timexes.groupby(['timex_ID'], return_type='spans') 
                
                a_event = None
                b_event = None
                for key, value in event_groups:
                    if key[0] == a_ID:
                        a_event = [span.base_span for span in value]
                    elif key[0] == b_ID:
                        b_event = [span.base_span for span in value]
            
                if a_event:
                    b_timex = None
                    for key, value in timex_groups:
                        if key[0] == b_ID:
                            b_timex = [span.base_span for span in value]
                            
                    tlinks_layer.add_annotation(a_text=a_event, a_ID=a_ID, rel_type=rel_type,
                                                b_text=b_timex, b_ID=b_ID, comment=comment)
                
                # if second argument is event, argument positions will be switched and relation type reversed
                elif b_event:
                    a_timex = None
                    for key, value in timex_groups:
                        if key[0] == a_ID:
                            a_timex = [span.base_span for span in value]               
                    if rel_type == 'AFTER':
                        rel_type = 'BEFORE'
                    elif rel_type == 'BEFORE':
                        rel_type = 'AFTER'
                    elif rel_type == 'INCLUDES':
                        rel_type = 'IS_INCLUDED'
                    elif rel_type == 'IS_INCLUDED':
                        rel_type = 'INCLUDES'
                
                    tlinks_layer.add_annotation(a_text=b_event, a_ID=b_ID, rel_type=rel_type,
                                                b_text=a_timex, b_ID=a_ID, comment=comment)
    
    return tlinks_layer


# -- method for creating event-DCT TLINK relation layer
def create_event_dct_rel_layer(text_obj, relations):
    tlinks_dct_layer = RelationLayer('event_dct_tlinks', span_names=['a_text'], 
                                     attributes=['a_ID', 'rel_type', 'b_ID', 'comment'], 
                                     display_order=['a_text', 'a_ID', 'rel_type', 'b_ID', 'comment'], 
                                     text_object=text_obj, enveloping='words')   
    last_ann = None   
    for entity in relations:
        ann = relations.get(entity)[0]
        if last_ann==ann:
            last_ann == None
            continue         
        else:
            last_ann = ann
            a_ID = ann[0]
            rel_type = ann[1]
            b_ID = ann[2]
            comment = ann[3]
            event_groups = text_obj.gold_events.groupby(['event_ID'], return_type='spans')
            
            a_event = None
            b_event = None
            for key, value in event_groups:
                if key[0] == a_ID:
                    a_event = [span.base_span for span in value]
                elif key[0] == b_ID:
                    b_event = [span.base_span for span in value]
            
            if a_event:
                tlinks_dct_layer.add_annotation(a_text=a_event, a_ID=a_ID, rel_type=rel_type,
                                        b_ID=b_ID, comment=comment)
            
            # if second argument is event, argument positions will be switched and relation type reversed
            elif b_event:
                if rel_type == 'AFTER':
                    rel_type = 'BEFORE'
                elif rel_type == 'BEFORE':
                    rel_type = 'AFTER'
                elif rel_type == 'INCLUDES':
                    rel_type = 'IS_INCLUDED'
                elif rel_type == 'IS_INCLUDED':
                    rel_type = 'INCLUDES'
                
                tlinks_dct_layer.add_annotation(a_text=b_event, a_ID=b_ID, rel_type=rel_type,
                                            b_ID=a_ID, comment=comment)
    
    # article DCT is added to relation layer metadata
    tlinks_dct_layer.meta['dct'] = text_obj.meta['dct']
    return tlinks_dct_layer


# -- method for creating EstNLTK Text-object from article text, returns Text-object and corpus word mapping
def create_Text_obj(sentence_base_annotations):
    letter_index = 0
    mapping = {}
    text_obj = ""
    
    for sentence_id, sentence in enumerate(sentence_base_annotations):
        sentence_text = []       
        for word_id_in_sentence, word_info in enumerate(sentence):
            word_text = word_info[2]
            letter_index += len(word_text)
            sentence_text.append(word_text)
            mapping[(sentence_id, word_id_in_sentence)] = (letter_index - len(word_text), letter_index)
            letter_index += 1
        sentence_text = " ".join(sentence_text)
        if text_obj == "":
            text_obj += sentence_text
        else:
            text_obj += " "+sentence_text
            
    return Text(text_obj).tag_layer('morph_analysis'), mapping


# -- main method for compiling and returning Text-object with layers
def create_Text_object_with_layers(filename, articlesDCT, baseAnnotations, eventsByLoc, timexesByLoc, event_timex_relations, event_dct_relations):
    sentence_base_annotations = baseAnnotations.get(filename)
    sentence_events = eventsByLoc.get(filename)
    sentence_timex = timexesByLoc.get(filename)
    event_timex_rels = event_timex_relations.get(filename)
    event_dct_rels = event_dct_relations.get(filename)
    # creating Text-object
    text_obj, mapping = create_Text_obj(sentence_base_annotations)
    text_obj.meta['dct'] = articlesDCT.get(filename)
    text_obj.meta['filename'] = filename
    # adding layers to Text-object
    event_layer, timex_layer = create_event_and_timex_layers(text_obj, sentence_base_annotations, sentence_events,
                                             sentence_timex, mapping)
    text_obj.add_layer( event_layer )
    text_obj.add_layer( timex_layer )
    text_obj.add_layer( create_gold_word_events_layer(text_obj) )
    text_obj.add_layer( create_gold_word_events_with_classes_layer(text_obj) )
    text_obj.add_layer( create_gold_event_phrases(text_obj) )
    text_obj.add_layer( create_gold_timex_phrases(text_obj) )
    text_obj.add_layer( create_event_timex_rel_layer(text_obj, event_timex_rels) )
    text_obj.add_layer( create_event_dct_rel_layer(text_obj, event_dct_rels) )
    
    return text_obj

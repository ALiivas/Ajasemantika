# Methods for converting article text to EstNLTK Text object and adding event and timex layers
#
# -- imports
import estnltk
from estnltk import Text


# -- method for creating event and timex layers
def create_layers(article_text, sentence_base_annotations, sentence_events, sentence_timex, mapping):
    # may be splitted into smaller methods later
    event_classes = ['REPORTING', 'PERCEPTION', 'ASPECTUAL', 'I_ACTION', 'I_STATE', 'STATE', 'MODAL',
                     'OCCURRENCE', 'EVENT_CONTAINER', 'CAUSE']
    timex_types = ['DATE', 'TIME', 'DURATION', 'SET']
    
    event_layer = estnltk.Layer(name='gold_events',
                  text_object=article_text,
                  attributes=['corpus_sentence_ID', 'corpus_word_ID','event_ID', 'expression', 'event_annotation',
                              'event_class'])
    timex_layer = estnltk.Layer(name='gold_timexes',
                  text_object=article_text,
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
            estnltk_word = article_text.words[estnltk_word_id]
            startend = mapping.get((sentence_id, word_id_in_sentence))
            estnltk_startend = tuple([estnltk_word.start, estnltk_word.end])  
                
            event = sentence_events.get((str(sentence_id), str(word_id_in_sentence)))
            if sentence_timex:
                timex = sentence_timex.get((str(sentence_id), str(word_id_in_sentence)))
            else:
                timex = None
            # Sõna on EVENT
            if event:
                # spanide mittekattuvuse korraL parandatakse sündmuse algus- ja lõpupositsioonid
                if estnltk_startend[0] <= startend[0] <= estnltk_startend[1] or startend[0] <= estnltk_startend[0] <= startend[1]:
                    #print(estnltk_word_id, "corpus_word:", word, startend, "EstNLTK_word:", estnltk_word.text, estnltk_startend)
                    startend = estnltk_startend
                    #print(estnltk_word_id, "corpus_word:", article_text.text[startend[0]:startend[1]], startend, "EstNLTK_word:", estnltk_word.text, estnltk_startend)
                    #print()
                event_tag = event[0][0]
                e_class = event[0][2].split()[1]
                # kui on multiword
                if 'multiword="true"' in event[0][2].split():
                    # multiword puhul on klass teisel kohal
                    if event[0][2].split()[1] not in event_classes:
                        # kui klassiga multiword osa oli tekstis eespool...
                        previous_class = None
                        for e in event_layer:
                            if e.event_ID==event_tag:
                                previous_class = e.event_annotation.split()[1]
                                break
                        # ...parandatakse klassi väärtus
                        if previous_class!=None and previous_class in event_classes:
                            e_class = previous_class
                        else:
                            previous_multiword = True
                    # kui klassita multiword osa oli tekstis eespool...
                    if previous_multiword:
                        for e in event_layer:
                            # ...parandatakse klassi väärtus
                            if e.event_ID==event_tag and e.event_class not in event_classes:
                                e.event_class=e_class
                                previous_multiword=False
                                break
                event_layer.add_annotation((startend[0], startend[1]), corpus_sentence_ID=sentence_id,
                                           corpus_word_ID=word_id_in_sentence, event_ID=event_tag,
                                           expression=event[0][1], event_annotation=event[0][2],
                                           event_class=e_class)
            # Sõna on TIMEX
            elif timex:
                # spanide mittekattuvuse korral parandatakse ajaväljendi algus- ja lõpupositsioonid
                if estnltk_startend[0] <= startend[0] <= estnltk_startend[1] or startend[0] <= estnltk_startend[0] <= startend[1]:
                    #print(estnltk_word_id, "corpus_word:", word, startend, "EstNLTK_word:", estnltk_word.text, estnltk_startend)
                    startend = estnltk_startend
                    #print(estnltk_word_id, "corpus_word:", article_text.text[startend[0]:startend[1]], startend, "EstNLTK_word:", estnltk_word.text, estnltk_startend)
                    #print()
                token_start = startend[0]
                token_end = startend[1]
                for i in range(len(timex)):
                    timex_type = timex[i][2].split()[1]
                    # Kui sellist timex_type'i pole, siis tõenäoliselt on tegemist multiword osaga
                    # Fraasi esimese sõna atribuutide väärtused kehtivad üle fraasi ka teistel sõnadel
                    if timex_type in timex_types:
                        last_timex_type = timex_type
                        timex_value = timex[i][2].split()[2]
                        last_timex_value = timex_value
                    else: # on multiword osa
                        timex_type = last_timex_type
                        timex_value = last_timex_value
                    # Tuleb starti ja endi muuta
                    if i > 0:
                        # wordi sees on expression
                        expression = timex[i][1].strip('""')
                        if word.find(expression) == -1:
                            if word == expression:
                                liita = token_start
                            else:
                                liita = len(timex[i-1][1].strip('""'))
                    timex_layer.add_annotation((token_start, token_end), corpus_sentence_ID=sentence_id,
                                               corpus_word_ID=word_id_in_sentence, timex_ID=timex[i][0],
                                               expression=timex[i][1], timex_annotation=timex[i][2],
                                               type=timex_type, value=timex_value)
                    
            if estnltk_word_id < len(article_text.words)-1:
                estnltk_word_id+=1
            
    return event_layer, timex_layer


# -- method for creating gold_word_events layer with IOB-annotations
def create_gold_word_events_layer(article_text, sentence_base_annotations):
    gold_word_events = estnltk.Layer(name="gold_word_events", text_object=article_text, attributes=['nertag'],
                                     enveloping='words')
    last_event_tag = None
    multiword_event_tags = set()
    for i, word in enumerate(article_text.words):
        event = article_text.gold_events.get(word)
        if event:
            is_multiword = 'multiword="true"' in event['event_annotation']
            event_tag = event['event_ID']
            if event_tag != last_event_tag and event_tag not in multiword_event_tags:
                gold_word_events.add_annotation([word.base_span], nertag="B-EVENT")
            else:
                gold_word_events.add_annotation([word.base_span], nertag="I-EVENT")
        
            if is_multiword:
                multiword_event_tags.add(event_tag)

            last_event_tag = event_tag
        else:
            gold_word_events.add_annotation([word.base_span], nertag="O")
            
    return gold_word_events


# -- method for creating EstNLTK Text object from article text, returns Text object and corpus word mapping
def create_article_text(sentence_base_annotations):
    letter_index = 0
    mapping = {}
    article_text = ""
    
    for sentence_id, sentence in enumerate(sentence_base_annotations):
        sentence_text = []
        
        for word_id_in_sentence, word_info in enumerate(sentence):
            word_text = word_info[2]
            letter_index += len(word_text)
            sentence_text.append(word_text)
            mapping[(sentence_id, word_id_in_sentence)] = (letter_index - len(word_text), letter_index)
            letter_index += 1

        sentence_text = " ".join(sentence_text)
        if article_text == "":
            article_text += sentence_text
        else:
            article_text += " "+sentence_text
            
    return Text(article_text).tag_layer('morph_analysis'), mapping


# -- method for adding layers and metadata to article text
def add_layers_to_text(filename, article_text, articlesDCT, layers_as_list): 
    article_text.meta['dct'] = articlesDCT.get(filename)
    article_text.meta['filename'] = filename
    for layer in layers_as_list:
        article_text.add_layer(layer)

    return article_text


# -- main method for compiling and returning Text object with layers
def create_Text_object(filename, articlesDCT, baseAnnotations, eventsByLoc, timexesByLoc):
    sentence_base_annotations = baseAnnotations.get(filename)
    sentence_events = eventsByLoc.get(filename)
    sentence_timex = timexesByLoc.get(filename)
    article_text, mapping = create_article_text(sentence_base_annotations)
    event_layer, timex_layer = create_layers(article_text, sentence_base_annotations, sentence_events,
                                                                   sentence_timex, mapping)
    article_text = add_layers_to_text(filename, article_text, articlesDCT, [event_layer, timex_layer])
    
    return article_text, mapping

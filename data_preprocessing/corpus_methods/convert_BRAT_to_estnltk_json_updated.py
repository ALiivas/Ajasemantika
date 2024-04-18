# ====================================================================
#  Utilities for converting Brat annotation files (*.txt, *.ann) to 
#  EstNLTK v1.6/v1.7 Text objects
#
#  Usage example:
#   python convert_BRAT_to_estnltk_json  [input_folder]  [output_folder] 
#
#  [input_folder] -- folder containing brat files (*.txt, *.ann);
#  [output_folder] -- folder for placing EstNLTK json files 
#                     (must be an existing folder);
#
#  Requirements
#     Python 3.7+
#     EstNLTK v1.6.9+
# ====================================================================

# Updated version of convert_BRAT_to_estnltk_json.py (accessible here: https://github.com/soras/EstTimexCorpora/blob/master/scripts/convert_BRAT_to_estnltk_json.py)

import os, os.path
import re
import sys

from estnltk import Text, Layer
from estnltk.converters import text_to_json
from estnltk_core import RelationLayer

# ====================================================================
#    Utilities for parsing Brat annotations                           
# ====================================================================

# Pattern for capturing names & values of attributes
tag_attribs_pat = re.compile("([^= ]+)='([^']+?)'")

def parse_tag_attributes( tag_str ):
    """Extracts names & values of attributes from an XML tag string,
       and returns as a dictionary."""
    assert tag_str.count("'") % 2 == 0, \
        '(!) Uneven number of quotation marks in: '+str(tag_str)
    attribs = {}
    for attr_match in tag_attribs_pat.finditer(tag_str):
        key   = attr_match.group(1)
        value = attr_match.group(2)
        if key in attribs:
           if attribs[key] != value:
               raise Exception(' (!) Unexpected: attribute "'+key+'" appears more than once with conflicting values in: '+tag_str)
        attribs[key] = value
    return attribs

simple_entity_pat = re.compile('^(T[0-9]+)\t+(\S+) ([0-9]+) ([0-9]+)\t(.+)$')

def _parse_entity_annotation( line ):
    #  Simple entity
    # Examples:
    # T1	Timex 0 13	Üleeilne päev
    # T2	Timex 128 138	sel päeval
    # T15	Event 120 127	leidsid
    # T16	Event 621 631	kingitakse
    m = simple_entity_pat.match(line)
    if m:
        entity_id    = m.group(1)
        entity_type  = m.group(2)
        entity_start = int(m.group(3))
        entity_end   = int(m.group(4))
        entity_text  = m.group(5)
        entity_attribs = {}
        entity_attribs['text'] = entity_text
        entity_attribs['_is_multiword'] = False
        return (entity_id, entity_type, entity_start, entity_end, entity_attribs)
    else:
        #
        # Multiword entity. Examples:
        #
        #  T22\tEvent 1786 1795;1796 1805\tpikendada tellimust
        #  T16\tEvent 1315 1318;1329 1333;1334 1337\toli kõne all
        #
        if line.count(';') == 0:
            raise ValueError( '(!) Unexpected entity annotation line: {!r}'.format(line) )
        items = line.split('\t')
        entity_id = items[0]
        entity_type_and_locs = items[1].split(' ')
        entity_texts = items[2].split(' ')
        entity_type = entity_type_and_locs[0]
        assert not entity_type.isnumeric(), '(!) Unexpcted entity type {!r}'.format(entity_type)
        raw_locs = []
        for loc_str in entity_type_and_locs[1:]:
            if loc_str.isnumeric():
                raw_locs.append( int(loc_str) )
            elif ';' in loc_str:
                for loc_str_2 in loc_str.split(';'):
                    raw_locs.append( int(loc_str_2) )
            else:
                raise ValueError( '(!) Unexpected entity loc: {!r}'.format(loc_str) )
        assert len(raw_locs) % 2 == 0, \
             '(!) Uneven number of raw locations in: '+str(raw_locs)
        entity_start = [loc for lid, loc in enumerate(raw_locs) if lid % 2 == 0]
        entity_end   = [loc for lid, loc in enumerate(raw_locs) if lid % 2 != 0]
        entity_attribs = {}
        entity_attribs['text'] = entity_texts
        entity_attribs['_is_multiword'] = True
        if len(entity_start) != len(entity_texts):
            print('(!) different number of entity texts {!r} and start locs {!r}'.format(entity_texts, entity_start) )
        return (entity_id, entity_type, entity_start, entity_end, entity_attribs)
    raise ValueError( '(!) Unexpected entity annotation line: {!r}'.format(line) )


attrib_annotation_specific_pat = re.compile('^(A[0-9]+)\t(\S+) (T[0-9]+) (\S+)$')

def _parse_attrib_annotation( line ):
    #  Examples:
    # A1	type T1 DATE
    # A2	type T2 DATE
    # A14	duration T14 hours
    # A15	duration_confidence T14 high
    # A16	class_confidence T14 high
    # A17	class T14 STATE
    m = attrib_annotation_specific_pat.match( line )
    if m:
        a_id    = m.group(1)
        a_name  = m.group(2)
        entity  = m.group(3)
        a_value = m.group(4)
        return (entity, a_name, a_value, a_id)
    else:
        raise ValueError( '(!) Unexpected attribute annotation line: {!r}'.format(line) )

def _add_attribute_annotations_to_entity_annotations( entity_annotations, attr_annotations ):
    for (entity_id1, a_name, a_value, a_id) in attr_annotations:
        for (entity_id2, entity_type, entity_start, entity_end, entity_attribs) in entity_annotations:
            if entity_id1 == entity_id2:
                if a_name in entity_attribs.keys():
                    if entity_attribs[a_name] != a_value:
                        raise ValueError('(!) conflicting values for entity attribute {!r}: {!r} vs {!r}'.format(a_name, entity_attribs[a_name], a_value))
                entity_attribs[a_name] = a_value

annotator_notes_specific_pat  = re.compile('^(#[0-9]+)\tAnnotatorNotes (\S+)\tOriginal: (.+)$')
annotator_notes_subtimex_pat  = re.compile('((part_of_interval|begin_point|end_point)=\{[^}]+\})')
annotator_notes_cutomized_pat = re.compile('^(#[0-9]+)\tAnnotatorNotes (\S+)\t(.+)$')

def _parse_notes_annotation( line ):
    #  Examples:
    # #1	AnnotatorNotes T1	Original: <TIMEX text='Üleeilne päev' tid='t1_1' type='DATE' value='1998-06-10' temporal_function=True>
    # #2	AnnotatorNotes T2	Original: <TIMEX text='sel päeval' tid='t1' type='DATE' value='1998-06-10' temporal_function=True anchor_time_id='t1'>
    m1 = annotator_notes_specific_pat.match( line )
    if m1:
        entity_id = m1.group(2)
        note_content = m1.group(3)
        if annotator_notes_subtimex_pat.search( note_content ):
            #
            #  We have to handle part_of_interval|begin_point|end_point subtimexes, like in these examples:
            #
            #   #11   AnnotatorNotes T11      Original: <TIMEX text='10.-' tid='t10' type='DATE' value='2001-09-10' temporal_function=True part_of_interval={'tid': 't12', 'type': 'DURATION', 'value': 'PXXD', 'temporal_function': True, 'begin_point': 't10', 'end_point': 't11'}>
            #   #1 AnnotatorNotes T1       Original: <TIMEX text='Viimase poolsajandi jooksul' tid='t1_1' type='DURATION' value='P50Y' temporal_function=True begin_point={'tid': 't1_2', 'type': 'DATE', 'value': 'XXX', 'temporal_function': True} end_point='t0'>
            # 
            #  The current solution is just to remove them.
            # 
            for match in annotator_notes_subtimex_pat.finditer( note_content ):
                s = match.start()
                e = match.end()
                note_content = note_content.replace(note_content[s:e], '')
        timex_tag_attribs = parse_tag_attributes( note_content )
        return (entity_id, timex_tag_attribs, 'TIMEX_ATTRIBS')
    m2 = annotator_notes_cutomized_pat.match( line )
    if m2:
        entity_id = m2.group(2)
        note_content = m2.group(3)
        return (entity_id, note_content, 'COMMENT')
    else:
        raise ValueError( '(!) Unexpected annotation notes line: {!r}'.format(line) )

tlink_relation_annotation_specific_pat = re.compile('^(R[0-9]+)\tTlink_(\S+) Arg1:(\S+) Arg2:(\S+)\t.*$')
has_argument_annotation_specific_pat = re.compile('^(R[0-9]+)\thas_Argument Arg1:(\S+) Arg2:(\S+)\t.*$')

def _parse_relation_annotation( line ):
    #  Tlink examples:
    # R1	Tlink_SIMULTANEOUS Arg1:T1 Arg2:T14	
    # R2	Tlink_INCLUDES Arg1:T2 Arg2:T15	
    # R3	Tlink_SIMULTANEOUS Arg1:T3 Arg2:T16	
    #  has_argument examples:
    # R10	has_Argument Arg1:T11 Arg2:T12	
    # R11	has_Argument Arg1:T15 Arg2:T16
    m1 = tlink_relation_annotation_specific_pat.match( line )
    if m1:
        rel_id   = m1.group(1)
        rel_type = m1.group(2)
        rel_arg1 = m1.group(3)
        rel_arg2 = m1.group(4)
        return (rel_arg1, rel_type, rel_arg2, rel_id)
    m2 = has_argument_annotation_specific_pat.match( line )
    if m2:
        rel_id   = m2.group(1)
        rel_type = 'has_Argument'
        rel_arg1 = m2.group(2)
        rel_arg2 = m2.group(3)
        return (rel_arg1, rel_type, rel_arg2, rel_id)
    else:
        raise ValueError( '(!) Unexpected relation annotation line: {!r}'.format(line) )

def import_brat_annotations( fname ):
    assert fname.endswith('.ann')
    annotations = []
    # 1) collect annotations
    entity_annotation_pat  = re.compile('^T[0-9]+\t.*')
    attrib_annotation_pat = re.compile('^A[0-9]+\t.*')
    notes_annotation_pat  = re.compile('^#[0-9]+\t+AnnotatorNotes.*')
    relation_annotation_pat = re.compile('^R[0-9]+\t.*')
    entity_annotations = []
    attr_annotations = []
    rel_annotations = []
    notes_annotations = []
    with open( fname, 'r', encoding='utf-8' ) as in_f:
        split_lines_ahead = 0
        for line in in_f:
            line = line.rstrip('\n')
            if len(line) == 0:
                continue
            if entity_annotation_pat.match(line):
                entity_annotations.append( _parse_entity_annotation( line ) )
            elif attrib_annotation_pat.match(line):
                attr_annotations.append( _parse_attrib_annotation( line ) )
            elif notes_annotation_pat.match(line):
                notes_annotations.append( _parse_notes_annotation( line ) )
            elif relation_annotation_pat.match(line):
                rel_annotations.append( _parse_relation_annotation( line ) )
            else:
                print('(!) Cannot parse annotation {!r}'.format(line))
    # 2) merge attribute annotations into entity annotations
    for (entity_id1, a_name, a_value, a_id) in attr_annotations:
        entity_found = False
        for (entity_id2, entity_type, entity_start, entity_end, entity_attribs) in entity_annotations:
            if entity_id1 == entity_id2:
                if a_name in entity_attribs.keys():
                    if entity_attribs[a_name] != a_value:
                        raise ValueError( ('(!) conflicting values for entity attribute {!r}: {!r} vs {!r}'+
                                           '').format( a_name, entity_attribs[a_name], a_value ) )
                entity_attribs[a_name] = a_value
                entity_found = True
                break
        if not entity_found:
            print( '(!) Cannot find entity {!r} to add attribute value {!r}'.format(entity_id1, {a_name:a_value}) )
    # 3) merge annotation notes annotations into entity annotations
    for (entity_id1, notes_content, notes_type) in notes_annotations:
        if notes_type == 'TIMEX_ATTRIBS':
            timex_tag_attribs = notes_content
            entity_found = False
            for (entity_id2, entity_type, entity_start, entity_end, entity_attribs) in entity_annotations:
                if entity_id1 == entity_id2:
                    for a_name, a_value in timex_tag_attribs.items():
                        if a_name in entity_attribs and entity_attribs[a_name] != a_value:
                            raise ValueError( ('(!) conflicting values for entity attribute {!r}: {!r} vs {!r}'+
                                               '').format( a_name, entity_attribs[a_name], a_value ) )
                        entity_attribs[a_name] = a_value
                    entity_found = True
                    break
            if not entity_found:
                print( '(!) Cannot find entity {!r} to add attribute values {!r}'.format(entity_id1, timex_tag_attribs) )
        elif notes_type == 'COMMENT':
            entity_found = False
            for (entity_id2, entity_type, entity_start, entity_end, entity_attribs) in entity_annotations:
                if entity_id1 == entity_id2:
                    if 'comment' not in entity_attribs:
                        entity_attribs['comment'] = notes_content
                    else:
                        entity_attribs['comment'] += ' | '+notes_content
                    entity_found = True
                    break
            if not entity_found:
                print( '(!) Cannot find entity {!r} to add comment {!r}'.format(entity_id1, notes_content) )
        else:
            raise Exception('(!) Unexpected AnnotationNotes {!r}'.format( (entity_id1, notes_content, notes_type) ) )
    return [entity_annotations, rel_annotations]

def import_brat_text( fname ):
    # Fetch textual content
    assert fname.endswith('.txt')
    content = None
    with open(fname, 'r', encoding='utf-8') as in_f:
        content = in_f.read()
    # Trick: convert every newline to double space
    # ( it seems that brat is doing something similar while indexing entities )
    #content = content.replace('\n', '  ')
    return content

def _calculate_corrected_start_and_delta( text, start ):
    corrected_start = start
    delta = 0
    i = 0
    while i < corrected_start:
        symbol = text[i]
        if symbol == '\n':
            # Substract 1 for every newline to get the correct location
            # ( brat seems to apply the same logic while 
            #   calculating annotations )
            corrected_start -= 1
            delta -= 1
        i += 1
    return corrected_start, delta

def import_from_brat_folder( folder ):
    assert os.path.isdir( folder ), \
        "(!) Invalid folder name {!r}.".format(folder)
    annotation_files = dict()
    for fname in os.listdir( folder ):
        if fname.endswith( ('.ann', '.txt') ):
            name, ext = os.path.splitext(fname)
            fpath = os.path.join( folder, fname )
            if name not in annotation_files:
                annotation_files[name] = []
            annotation_files[name].append( fpath )
    # Check that both .ann and .txt exist
    for name in annotation_files.keys():
        if len( annotation_files[name] ) != 2:
            has_ann = any([ fname for fname in annotation_files[name] if fname.endswith('.ann') ])
            has_txt = any([ fname for fname in annotation_files[name] if fname.endswith('.txt') ])
            if not has_txt:
                raise ValueError('(!) Annotations file {!r} is missing .txt part.'.format(name))
            if not has_ann:
                raise ValueError('(!) Annotations file {!r} is missing .ann part.'.format(name))
    text_objects = []
    for name in annotation_files.keys():
        ann_file = [fname for fname in annotation_files[name] if fname.endswith('.ann')][0]
        entity_annotations, rel_annotations = import_brat_annotations( ann_file )
        txt_file = [fname for fname in annotation_files[name] if fname.endswith('.txt')][0]
        content = import_brat_text( txt_file )
        #
        #  Create text object and entity annotations
        #
        text_obj = Text(content).tag_layer('morph_analysis')
        text_obj.meta['filename'] = name  
        event_layer = \
            Layer('events', attributes=('brat_id', 'class', 'class_confidence', 'duration', 'duration_confidence', 'comment'), \
                            text_object = text_obj, enveloping='words')
        timex_layer = \
            Layer('timexes', attributes=('brat_id', 'tid', 'type', 'value', 'mod', 'anchor_time_id', 'comment'), \
                             text_object = text_obj, enveloping='words', ambiguous=True)
        entity_layer = \
            Layer('entities', attributes=('brat_id',), text_object = text_obj, enveloping='words')
        entity_id_to_loc_map = dict()
        for (entity_id, type, start, end, attribs) in entity_annotations:
            # Check that location strings are expected ones
            # Collect corrected locations
            corrected_locs = []
            if isinstance(start, int):
                corrected_start, delta = _calculate_corrected_start_and_delta( content, start )
                snippet = content[corrected_start : end+delta]
                assert snippet == attribs['text'], \
                    f"(!) {name!r} has mismatching entity texts {snippet!r} vs {attribs['text']!r}"
                corrected_locs.append( (corrected_start, end+delta) )
            elif isinstance(start, list):
                if len(start) == len(attribs['text']):
                    for s_start, s_end, s_text in zip(start, end, attribs['text']):
                        corrected_start, delta = _calculate_corrected_start_and_delta( content, s_start )
                        snippet = content[corrected_start : s_end+delta]
                        assert snippet == s_text, \
                            f"(!) {name!r} has mismatching entity texts {snippet!r} vs {attribs['text']!r}"
                        corrected_locs.append( (corrected_start, s_end+delta) )
                elif len(start) <= len(attribs['text']):
                    # Tricky case: there can be less entity locations than entity text strings
                    # (!) different number of entity texts ['oli', 'kõige', 'parem'] and start locs [1904, 1908]
                    assert len(start) == len(end)
                    for s_start, s_end in zip(start, end):
                        corrected_start, delta = _calculate_corrected_start_and_delta( content, s_start )
                        snippet = content[corrected_start : s_end+delta]
                        assert any([s in snippet for s in attribs['text']]), \
                            f"(!) {name!r} has mismatching entity texts {snippet!r} vs {attribs['text']!r}"
                        corrected_locs.append( (corrected_start, s_end+delta) )
                else:
                    raise Exception('(!) Mismatching number of locations and texts in {!r}'.format( (entity_id, type, start, end, attribs) ) )
            entity_id_to_loc_map[entity_id] = corrected_locs
            #
            # Synchronize corpus tokenization with EstNLTK word tokenization
            #
            estnltk_words = []
            for loc in corrected_locs:
                for i in range(len(text_obj.words)):
                    estnltk_startend = tuple([text_obj.words[i].start, text_obj.words[i].end])
                    if estnltk_startend == loc:
                        # 1) full overlap
                        estnltk_words.append(text_obj.words[i])
                    elif estnltk_startend[0] <= loc[0] and loc[1] <= estnltk_startend[1]:
                        # 2) partial overlap: EstNLTK word contains corpus word
                        estnltk_words.append(text_obj.words[i])
                    elif loc[0] <= estnltk_startend[0] and estnltk_startend[1] <= loc[1]:
                        # 3) partial overlap: corpus word contains EstNLTK word
                        estnltk_words.append(text_obj.words[i])
                    elif estnltk_startend[0] < loc[0] < estnltk_startend[1]:
                        # 4) partial overlap: EstNLTK word starts corpus word (but might not end it)
                        estnltk_words.append(text_obj.words[i])
                    elif loc[0] < estnltk_startend[0] < loc[1]:
                        # 5) partial overlap: corpus word starts EstNLTK word (but might not end it)
                        estnltk_words.append(text_obj.words[i])
            # add enveloping layers
            if type == 'Event':
                attribs['brat_id'] = entity_id
                event_layer.add_annotation( [s.base_span for s in estnltk_words], **attribs )
            elif type == 'Timex':
                attribs['brat_id'] = entity_id
                timex_layer.add_annotation( [s.base_span for s in estnltk_words], **attribs )
            elif type == 'Entity':
                attribs['brat_id'] = entity_id
                entity_layer.add_annotation( [s.base_span for s in estnltk_words], **attribs )
        text_obj.add_layer( event_layer )
        text_obj.add_layer( timex_layer )
        text_obj.add_layer( entity_layer )
        #
        # Add event layer with IOB-annotations
        #
        gold_word_events_layer = Layer(name="gold_word_events", text_object=text_obj, 
                                 attributes=['nertag'], enveloping='words')  
        for word in text_obj.words:
            # searching event layer for current word
            event_idx = None
            for event in text_obj.events:
                for j in range(len(event)):
                    if event[j] == word:
                        event_idx = j
                        break
            if event_idx != None:
                # if word is a single-word event or is first word of a multiword event, its tag will be 'B-EVENT'
                if event_idx == 0:
                    gold_word_events_layer.add_annotation([word.base_span], nertag="B-EVENT")
                # if word is any subsequent word of a multiword event, its tag will be 'I-EVENT'
                else:
                    gold_word_events_layer.add_annotation([word.base_span], nertag="I-EVENT")
            # if word is not part of any event, its tag will be 'O'
            else:
                gold_word_events_layer.add_annotation([word.base_span], nertag="O")        
        text_obj.add_layer( gold_word_events_layer )   
        #
        #  Add tlink relation annotations
        #
        relations_layer = RelationLayer('tlinks', span_names=['a_text', 'b_text'], 
                                 attributes=['brat_id', 'rel_type'], 
                                 display_order=['brat_id', 'a_text', 'rel_type', 'b_text'], 
                                 text_object=text_obj, enveloping='words', ambiguous=True)  
        for (rel_arg1, rel_type, rel_arg2, rel_id) in rel_annotations:
            if rel_type == 'has_Argument':
                continue
            assert rel_arg1 in entity_id_to_loc_map.keys()
            assert rel_arg2 in entity_id_to_loc_map.keys()
            arg1_event = [text_obj['events'][i] for i in range(len(text_obj['events'])) if text_obj['events'][i].brat_id==rel_arg1]
            arg2_event = [text_obj['events'][i] for i in range(len(text_obj['events'])) if text_obj['events'][i].brat_id==rel_arg2]
            arg1_entity = [text_obj['entities'][i] for i in range(len(text_obj['entities'])) if text_obj['entities'][i].brat_id==rel_arg1]
            arg2_entity = [text_obj['entities'][i] for i in range(len(text_obj['entities'])) if text_obj['entities'][i].brat_id==rel_arg2]
            if arg1_event or arg1_entity:
                arg2_timex = None
                for i in range(len(text_obj['timexes'])):
                    for j in range(len(text_obj['timexes'][i].annotations)):
                        if text_obj['timexes'][i].annotations[j]['brat_id'] == rel_arg2:
                            arg1_timex = text_obj['timexes'][i].annotations[j]
                event_or_entity = arg1_event or arg1_entity
                relations_layer.add_annotation(brat_id=rel_id, a_text=[s.base_span for s in event_or_entity[0]], 
                                                rel_type=rel_type, 
                                                b_text=[s.base_span for s in arg2_timex[0]])
            # check if relation needs to be reversed
            # relation will be reversed, if event/entity is second argument
            elif arg2_event or arg2_entity:
                arg1_timex = None
                for i in range(len(text_obj['timexes'])):
                    for j in range(len(text_obj['timexes'][i].annotations)):
                        if text_obj['timexes'][i].annotations[j]['brat_id'] == rel_arg1:
                            arg1_timex = text_obj['timexes'][i].annotations[j]
                event_or_entity = arg2_event or arg2_entity
                if len(arg1_timex)==0:
                    print(arg1_timex, rel_arg1, text_obj.meta['file'])
                if rel_type == 'AFTER':
                    rel_type = 'BEFORE'
                elif rel_type == 'BEFORE':
                    rel_type = 'AFTER'
                elif rel_type == 'INCLUDES':
                    rel_type = 'IS_INCLUDED'
                elif rel_type == 'IS_INCLUDED':
                    rel_type = 'INCLUDES'
                relations_layer.add_annotation(brat_id=rel_id, a_text=[s.base_span for s in event_or_entity[0]], 
                                                rel_type=rel_type, 
                                                b_text=[s.base_span for s in arg1_timex.span])        
        text_obj.add_layer( relations_layer )
        #
        #  Add has_Argument relations
        #
        arguments_layer = RelationLayer('event_arguments', span_names=['a_text', 'b_text'], 
                                 attributes=['brat_id', 'rel_type'], 
                                 display_order=['brat_id', 'a_text', 'rel_type', 'b_text'], 
                                 text_object=text_obj, enveloping='words', ambiguous=True)
        for (rel_arg1, rel_type, rel_arg2, rel_id) in rel_annotations:
            if rel_type != 'has_Argument':
                continue 
            assert rel_arg1 in entity_id_to_loc_map.keys()
            assert rel_arg2 in entity_id_to_loc_map.keys()
            arg1_loc = entity_id_to_loc_map[rel_arg1]
            arg2_loc = entity_id_to_loc_map[rel_arg2]  
            arg1_event = [text_obj['events'][i] for i in range(len(text_obj['events'])) if text_obj['events'][i].brat_id==rel_arg1]
            arg2_event = [text_obj['events'][i] for i in range(len(text_obj['events'])) if text_obj['events'][i].brat_id==rel_arg2]
            # check if relation needs to be reversed
            if arg1_loc[0] > arg2_loc[0]:
                # reverse relation
                temp = arg1_event
                arg1_event = arg2_event
                arg2_event = temp
                # change rel_type        
                rel_type = 'is_Argument_of'    
            arguments_layer.add_annotation(brat_id=rel_id, a_text=[s.base_span for s in arg1_event[0]], 
                                                rel_type=rel_type, 
                                                b_text=[s.base_span for s in arg2_event[0]])
        text_obj.add_layer( arguments_layer )
        text_objects.append( text_obj )
    return text_objects

if __name__ == '__main__':
    input_folder  = None
    output_folder = None
    if len(sys.argv) > 2:
        input_folder = sys.argv[1]
        assert os.path.isdir(input_folder), \
            '(!) Unexpected input folder: {!r}. Please give name of the input folder as the first argument.'.format(input_folder)
        output_folder = sys.argv[2]
        assert os.path.isdir(output_folder), \
            '(!) Unexpected output folder: {!r}. Please give name of the (existing) output folder as the second argument.'.format(output_folder)
        text_objects = import_from_brat_folder( input_folder )
        for text in text_objects:
            fpath = os.path.join( output_folder, text.meta['file']+'.json' )
            print('=>', fpath)
            text_to_json(text, file=fpath)
        print(f"{len(text_objects)} files converted.")
    else:
        print(f'(!) Missing command line arguments input_folder and output_folder.\n'+\
              f'Usage:  python  {sys.argv[0]}  [input_folder]  [output_folder] ')

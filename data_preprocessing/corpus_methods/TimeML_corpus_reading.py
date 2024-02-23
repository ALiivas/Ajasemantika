# Methods for corpus reading from https://github.com/soras/EstTimeMLCorpus/blob/master/exported_corpus_reader.py
#
# -- imports
import re

# -- files
baseAnnotationFile     = "EstTimeMLCorpus/corpus/base-segmentation-morph-syntax"
eventAnnotationFile    = "EstTimeMLCorpus/corpus/event-annotation"
timexAnnotationFile    = "EstTimeMLCorpus/corpus/timex-annotation"
timexAnnotationDCTFile = "EstTimeMLCorpus/corpus/timex-annotation-dct"
tlinkEventTimexFile    = "EstTimeMLCorpus/corpus/tlink-event-timex"
tlinkEventDCTFile      = "EstTimeMLCorpus/corpus/tlink-event-dct"
tlinkMainEventsFile    = "EstTimeMLCorpus/corpus/tlink-main-events"
tlinkSubEventsFile     = "EstTimeMLCorpus/corpus/tlink-subordinate-events"
articleMetadata_file   = "EstTimeMLCorpus/corpus/article-metadata"


def load_base_segmentation(inputFile):
    base_segmentation = dict()
    last_sentenceID = ""
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if ( re.match("^#.+$", line) ):
            continue
        items = (line.rstrip()).split("\t")
        if (len(items) != 7):
            raise Exception(" Unexpected number of items on line: '"+str(line)+"'")
        file = items[0]
        if (file not in base_segmentation):
            base_segmentation[file] = []
        sentenceID = items[1]
        if (sentenceID != last_sentenceID):
            base_segmentation[file].append([])
        wordID     = items[2]
        # fileName	sentence_ID	word_ID_in_sentence	token	morphological_and_syntactic_annotations	syntactic_ID	syntactic_ID_of_head
        token           = items[3]
        morphSyntactic  = items[4]
        syntacticID     = items[5]
        syntacticHeadID = items[6]
        base_segmentation[file][-1].append( [sentenceID, wordID, token, morphSyntactic, syntacticID, syntacticHeadID] )
        last_sentenceID = sentenceID
    f.close()
    return base_segmentation
    
def load_entity_annotation(inputFile):
    annotationsByLoc = dict()
    annotationsByID  = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if ( re.match("^#.+$", line) ):
            continue
        items = (line.rstrip()).split("\t")
        # fileName	sentence_ID	word_ID_in_sentence	expression	event_annotation	event_ID
        # fileName	sentence_ID	word_ID_in_sentence	expression	timex_annotation	timex_ID
        if (len(items) != 6):
            raise Exception(" Unexpected number of items on line: '"+str(line)+"'")
        file       = items[0]
        sentenceID = items[1]
        wordID     = items[2]
        expression = items[3]
        annotation = items[4]
        entityID   = items[5]
        if (file not in annotationsByLoc):
            annotationsByLoc[file] = dict()
        if (file not in annotationsByID):
            annotationsByID[file] = dict()
        # Record annotation by its location in text
        locKey = (sentenceID, wordID)
        if (locKey not in annotationsByLoc[file]):
            annotationsByLoc[file][locKey] = []
        annotationsByLoc[file][locKey].append( [entityID, expression, annotation] )
        # Record annotation by its unique ID in text
        if (entityID not in annotationsByID[file]):
            annotationsByID[file][entityID] = []
        annotationsByID[file][entityID].append( [sentenceID, wordID, expression, annotation] )
    f.close()
    return (annotationsByLoc, annotationsByID)

def load_dct_annotation(inputFile):
    DCTsByFile = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if ( re.match("^#.+$", line) ):
            continue
        items = (line.rstrip()).split("\t")
        # fileName	document_creation_time
        if (len(items) != 2):
            raise Exception(" Unexpected number of items on line: '"+str(line)+"'")
        file = items[0]
        dct  = items[1]
        DCTsByFile[ file ] = dct
    f.close()
    return DCTsByFile

def load_relation_annotation(inputFile):
    annotationsByID  = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if ( re.match("^#.+$", line) ):
            continue
        items = line.split("\t")
        # old format: fileName	entityID_A	relation	entityID_B	comment	expression_A	expression_B
        # new format: fileName	entityID_A	relation	entityID_B	comment	
        if (len(items) != 5):
            print (len(items))
            raise Exception(" Unexpected number of items on line: '"+str(line)+"'")
        file     = items[0]
        entityA  = items[1]
        relation = items[2]
        entityB  = items[3]
        comment  = items[4].rstrip()
        if (file not in annotationsByID):
            annotationsByID[file] = dict()
        annotation = [entityA, relation, entityB, comment]
        if (entityA not in annotationsByID[file]):
            annotationsByID[file][entityA] = []
        annotationsByID[file][entityA].append( annotation )
        if (entityB not in annotationsByID[file]):
            annotationsByID[file][entityB] = []
        annotationsByID[file][entityB].append( annotation )
    f.close()
    return annotationsByID

def load_relation_to_dct_annotations(inputFile):
    annotationsByID  = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
        # Skip the comment line
        if ( re.match("^#.+$", line) ):
            continue
        items = line.split("\t")
        # old format: fileName	entityID_A	relation_to_DCT	comment	expression_A
        # new format: fileName	entityID_A	relation_to_DCT	comment
        if (len(items) != 4):
            raise Exception(" Unexpected number of items on line: '"+str(line)+"'")
        file          = items[0]
        entityA       = items[1]
        relationToDCT = items[2]
        comment       = items[3].rstrip()
        if (file not in annotationsByID):
            annotationsByID[file] = dict()
        annotation = [entityA, relationToDCT, "t0", comment]
        if (entityA not in annotationsByID[file]):
            annotationsByID[file][entityA] = []
        annotationsByID[file][entityA].append( annotation )
    f.close()
    return annotationsByID

# Method for getting article DCT (not from TimeMLCorpus)
def load_articles_DCT(inputFile):
    articlesDCT = dict()
    f = open(inputFile, mode='r', encoding="utf-8")
    for line in f:
    # Skip the comment line
        if ( re.match("^#.+$", line) ):
            continue
        items = line.split("\t")
        article = items[0]
        info = items[1].split(" | ")
        for inf in info:
            if "ajalehenumber" in inf:
                match = re.search(r'(\d+\.\d+\.\d+)',inf)
                #print(match.group(1))
                articlesDCT[article] = match.group(1)
                break
    return articlesDCT

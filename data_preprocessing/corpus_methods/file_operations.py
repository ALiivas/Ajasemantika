# Different methods for saving and loading corpus data

# -- imports
import os
import json
from estnltk.converters import text_to_json, json_to_text

# -- new target directory for EstTimeMLCorpus articles in JSON format
json_dir = 'EstTimeML_corpus_json'

if not os.path.isdir(json_dir):
    os.mkdir(json_dir)

# -- target folder path
json_dir_path = 'EstTimeML_corpus_json/'


# -- method for converting EstNLTK Text-object to JSON string and saving it in JSON format
def save_Text_to_json(text_object):
    #json_text = text_to_json(text_object)
    #with open(json_dir_path + text_object.meta['filename'] + '.json', 'w') as f:
    #    json.dump(json_text, f)
    filename = json_dir_path + text_object.meta['filename'] + '.json'
    text_to_json(text_object, file=filename)
        
# -- method for loading JSON string from JSON file and returning it as EstNLTK Text-object
def load_Text_from_json(filename):
    return json_to_text(file=json_dir_path + filename)
        
        
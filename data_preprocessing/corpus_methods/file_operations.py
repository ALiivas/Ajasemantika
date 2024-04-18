# Methods for saving and loading corpus data

# -- imports
import os
import json
from estnltk.converters import text_to_json, json_to_text


# -- method for converting EstNLTK Text-object to JSON string and saving it in JSON format
def save_Text_to_json(path, text_object):
    filename = path + text_object.meta['filename'] + '.json'
    text_to_json(text_object, file=filename)

    
# -- method for loading JSON string from JSON file and returning it as EstNLTK Text-object
def load_Text_from_json(path, filename):
    return json_to_text(file=path + filename)
        
        